# CUDA smoke test for the KernelAbstractions residual path (GPU_PLAN.md task 8).
#
# CUDA.jl is deliberately NOT a dependency of TwoDG; install it once into the
# shared "@cuda" environment so it's picked up via environment stacking:
#     julia +1.12 -e 'using Pkg; Pkg.activate("cuda"; shared=true); Pkg.add("CUDA")'
# then run (JULIA_LOAD_PATH separator is ';' on Windows, ':' elsewhere):
#     JULIA_LOAD_PATH='@;@cuda;@v#.#;@stdlib' julia +1.12 --project=. run_ka_cuda.jl
#
# Uses Float32 (consumer GPUs have weak FP64) and a moderate mesh (4 GB VRAM).

using TwoDG
using CUDA
using Adapt
using StaticArrays
using LinearAlgebra
using Printf

@assert CUDA.functional() "CUDA is not functional on this machine"
println("GPU: ", CUDA.name(CUDA.device()))

# ---- problem setup: Euler vortex-ish perturbation on a square mesh ----
γ = 1.4
uinf = [1.0, 0.3, 0.05, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2)]
n, porder = 33, 3
mesh = mkmesh_square(n, n, porder, 0, 1)
master = Master(mesh)

app = mkapp_euler_pt(; gamma=γ, bcm=fill(1, 4), bcs=reshape(uinf, 1, 4))

ρ(x, y) = 1.0 + 0.1 * exp(-30 * ((x - 0.5)^2 + (y - 0.5)^2))
u0 = initu(mesh, app, [ρ,
                       (x, y) -> 0.3 * ρ(x, y),
                       (x, y) -> 0.05 * ρ(x, y),
                       (x, y) -> 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2) * ρ(x, y)])

# ---- correctness: Float64 GPU must match Float64 CPU to roundoff ----
ctx64 = DGContext(master, mesh)
app_gpu = adapt(CuArray, app)
r64_cpu = rinvexpl_ka(ctx64, app, u0, 0.0)
r64_gpu = rinvexpl_ka(adapt(CuArray, ctx64), app_gpu, CuArray(u0), 0.0)
rel64 = norm(Array(r64_gpu) .- r64_cpu) / norm(r64_cpu)
@printf "Float64 GPU vs CPU residual relative difference: %.3e\n" rel64
@assert rel64 < 1e-10 "GPU/CPU mismatch in Float64 — kernel bug"

# ---- Float32 (the precision actually used for performance): differences are
# rounding noise (atomic ordering, FMA contraction) amplified by inverse-mass
# conditioning, so only a loose check is meaningful ----
ctx_cpu = DGContext(master, mesh; T=Float32)
u32 = Float32.(u0)
r_cpu = rinvexpl_ka(ctx_cpu, app, u32, 0.0f0)

ctx_gpu = adapt(CuArray, ctx_cpu)
u_gpu = CuArray(u32)
r_gpu = rinvexpl_ka(ctx_gpu, app_gpu, u_gpu, 0.0f0)

rel32_gpu = norm(Float64.(Array(r_gpu)) .- r64_cpu) / norm(r64_cpu)
rel32_cpu = norm(Float64.(r_cpu) .- r64_cpu) / norm(r64_cpu)
@printf "Float32 error vs Float64 reference: CPU %.3e, GPU %.3e\n" rel32_cpu rel32_gpu
@assert rel32_gpu < 10 * max(rel32_cpu, 1e-6) "GPU Float32 error far exceeds CPU Float32 error"
@assert all(isfinite, Array(r_gpu))

# ---- timing: repeated residual evaluations with preallocated workspace ----
nrep = 50
ws_cpu = RinvWorkspace(ctx_cpu, app.nc)
rr_cpu = similar(u32)
rinvexpl!(rr_cpu, ctx_cpu, app, u32, 0.0f0; ws=ws_cpu)  # warmup
t_cpu = @elapsed for _ in 1:nrep
    rinvexpl!(rr_cpu, ctx_cpu, app, u32, 0.0f0; ws=ws_cpu)
end

ws_gpu = RinvWorkspace(ctx_gpu, app.nc)
rr_gpu = similar(u_gpu)
rinvexpl!(rr_gpu, ctx_gpu, app_gpu, u_gpu, 0.0f0; ws=ws_gpu)  # warmup
t_gpu = @elapsed for _ in 1:nrep
    rinvexpl!(rr_gpu, ctx_gpu, app_gpu, u_gpu, 0.0f0; ws=ws_gpu)
end

nt = size(mesh.t, 1)
@printf "mesh: %d elements, porder %d, %d residual evals\n" nt porder nrep
@printf "CPU (KA backend, %d threads): %.3f s   GPU: %.3f s   speedup: %.1fx\n" Threads.nthreads() t_cpu t_gpu t_cpu / t_gpu

# ---- a few RK4 steps on the GPU ----
uu = copy(u_gpu)
rk4_ka!(ctx_gpu, app_gpu, uu, 0.0f0, 1f-4, 20; ws=ws_gpu)
@assert all(isfinite, Array(uu))
println("RK4 on GPU: OK")

# ============================ LDG viscous path ============================
println("\n-- LDG viscous path (convection-diffusion) --")

appv = mkapp_convection_diffusion_pt(x -> SVector(-x[2], x[1]);
                                     kappa=0.01, c11=10.0, c11int=0.5,
                                     bcm=[1, 2, 1, 2], bcs=zeros(2, 1))
appv_gpu = adapt(CuArray, appv)
uv0 = initu(mesh, appv, [(x, y) -> exp(-4 * ((x - 0.5)^2 + (y - 0.5)^2))])

# Float64 GPU vs CPU parity (gradient and residual)
qv_cpu = getq_ka(ctx64, appv, uv0, 0.0)
qv_gpu = getq_ka(adapt(CuArray, ctx64), appv_gpu, CuArray(uv0), 0.0)
relq = norm(Array(qv_gpu) .- qv_cpu) / norm(qv_cpu)
rv_cpu = rldgexpl_ka(ctx64, appv, uv0, 0.0)
rv_gpu = rldgexpl_ka(adapt(CuArray, ctx64), appv_gpu, CuArray(uv0), 0.0)
relv = norm(Array(rv_gpu) .- rv_cpu) / norm(rv_cpu)
@printf "Float64 GPU vs CPU: getq %.3e, rldgexpl %.3e\n" relq relv
@assert relq < 1e-10 && relv < 1e-10 "LDG GPU/CPU mismatch in Float64 — kernel bug"

# Float32 timing, CPU KA backend vs GPU
uv32 = Float32.(uv0)
wsv_cpu = RldgWorkspace(ctx_cpu, appv.nc)
rrv_cpu = similar(uv32)
rldgexpl!(rrv_cpu, ctx_cpu, appv, uv32, 0.0f0; ws=wsv_cpu)  # warmup
tv_cpu = @elapsed for _ in 1:nrep
    rldgexpl!(rrv_cpu, ctx_cpu, appv, uv32, 0.0f0; ws=wsv_cpu)
end

wsv_gpu = RldgWorkspace(ctx_gpu, appv.nc)
uv_gpu = CuArray(uv32)
rrv_gpu = similar(uv_gpu)
rldgexpl!(rrv_gpu, ctx_gpu, appv_gpu, uv_gpu, 0.0f0; ws=wsv_gpu)  # warmup
tv_gpu = @elapsed for _ in 1:nrep
    rldgexpl!(rrv_gpu, ctx_gpu, appv_gpu, uv_gpu, 0.0f0; ws=wsv_gpu)
end
@printf "LDG residual, %d evals: CPU %.3f s   GPU %.3f s   speedup: %.1fx\n" nrep tv_cpu tv_gpu tv_cpu / tv_gpu

# a few viscous RK4 steps on the GPU
rk4_ka!(rldgexpl!, ctx_gpu, appv_gpu, uv_gpu, 0.0f0, 1f-4, 20; ws=wsv_gpu)
@assert all(isfinite, Array(uv_gpu))
println("LDG RK4 on GPU: OK")

# ==================== HDG trace solver (Phase 3) ====================
println("\n-- HDG KA/Krylov trace solver (convection-diffusion) --")
using TwoDG.HybridizableDiscontinuousGalerkin: hdg_elemmats

hdg_source(p) = reshape(2π^2 .* sin.(π .* p[:, 1]) .* sin.(π .* p[:, 2]), :, 1)
hdg_dbc(p) = zeros(size(p, 1), 1)
hdg_param = Dict(:kappa => 1.0, :c => [1.0, 0.5], :taud => 1.0)

nh, ph = 49, 3
mesh_h = mkmesh_square(nh, nh, ph, 0, 1)
master_h = Master(mesh_h, 4 * (ph + 1))
@printf "HDG mesh: %d elements, porder %d, %d trace DOFs\n" size(mesh_h.t, 1) ph size(mesh_h.f, 1) * (ph + 1)

print("CPU element assembly: ")
t_asm = @elapsed ae_h, fe_h = hdg_elemmats(master_h, mesh_h, hdg_source, hdg_dbc, hdg_param)
@printf "%.2f s\n" t_asm
sys_h = HDGSystem(ae_h, fe_h, mesh_h)

# Float64 GPU vs CPU: identical system, both must converge to the same solution
x_cpu, st_cpu = hdg_gmres_ka(sys_h; tol=1e-10, restart=200)
sys_gpu = adapt(CuArray, sys_h)
x_gpu, st_gpu = hdg_gmres_ka(sys_gpu; tol=1e-10, restart=200)
relh = norm(Array(x_gpu) .- x_cpu) / norm(x_cpu)
@printf "Float64 GPU vs CPU trace solution: %.3e (CPU %d / GPU %d iters)\n" relh st_cpu.niter st_gpu.niter
@assert relh < 1e-7 "HDG GPU/CPU mismatch in Float64"

# Float32 solve on the GPU
sys32_gpu = adapt(CuArray, HDGSystem(ae_h, fe_h, mesh_h; T=Float32))
x32_gpu, st32 = hdg_gmres_ka(sys32_gpu; tol=1e-5, restart=200)
rel32h = norm(Float64.(Array(x32_gpu)) .- x_cpu) / norm(x_cpu)
@printf "Float32 GPU trace solution vs Float64: %.3e (solved=%s)\n" rel32h st32.solved
@assert rel32h < 1e-3

# timing: GMRES solve only (assembly is CPU either way)
t_hcpu = @elapsed hdg_gmres_ka(sys_h; tol=1e-10, restart=200)
t_hgpu = @elapsed hdg_gmres_ka(sys_gpu; tol=1e-10, restart=200)
t_hgpu32 = @elapsed hdg_gmres_ka(sys32_gpu; tol=1e-5, restart=200)
@printf "GMRES: CPU F64 %.3f s   GPU F64 %.3f s (%.1fx)   GPU F32 %.3f s (%.1fx)\n" t_hcpu t_hgpu t_hcpu / t_hgpu t_hgpu32 t_hcpu / t_hgpu32
println("HDG on GPU: OK")

# ---- batched assembly + recovery (Phase 3b) ----
println("\n-- HDG batched assembly + recovery (Phase 3b) --")

print("HDGBatch build (CPU geometry): ")
t_bb = @elapsed batch_h = HDGBatch(master_h, mesh_h, hdg_source, hdg_param)
@printf "%.2f s\n" t_bb

# CPU-backend batched assembly vs legacy elemmats (already have ae_h, fe_h with
# BCs applied — rebuild raw ones for the check via hdg_local_solves on CPU)
loc_cpu = hdg_local_solves(batch_h)
batch_gpu = adapt(CuArray, batch_h)
loc_gpu = hdg_local_solves(batch_gpu)
rel_ae = norm(Array(loc_gpu.ae) .- loc_cpu.ae) / norm(loc_cpu.ae)
@printf "batched ae, GPU vs CPU backend: %.3e\n" rel_ae
@assert rel_ae < 1e-10

t_bcpu = @elapsed hdg_local_solves(batch_h)
t_bgpu = @elapsed hdg_local_solves(batch_gpu)
@printf "batched local solves: CPU %.3f s   GPU %.3f s (%.1fx)   [legacy CPU assembly: %.2f s]\n" t_bcpu t_bgpu t_bcpu / t_bgpu t_asm

# end-to-end batched driver on GPU vs legacy CPU solution
u_gpu_b, q_gpu_b, _, _ = hdg_parsolve_batched(master_h, mesh_h, hdg_source, hdg_dbc, hdg_param;
                                              ArrayT=CuArray, tol=1e-10, restart=200)
u_ref, q_ref, _, _ = hdg_parsolve(master_h, mesh_h, hdg_source, hdg_dbc, hdg_param;
                                  tol=1e-10, restart=200)
rel_u = norm(u_gpu_b .- u_ref) / norm(u_ref)
rel_q = norm(q_gpu_b .- q_ref) / norm(q_ref)
@printf "end-to-end GPU batched vs legacy CPU: u %.3e, q %.3e\n" rel_u rel_q
@assert rel_u < 1e-7 && rel_q < 1e-7
println("HDG batched on GPU: OK")
