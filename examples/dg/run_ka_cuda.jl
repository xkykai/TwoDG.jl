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

phys = DGPhysics(EulerEquations(γ=γ); boundary_conditions=ntuple(_ -> FarField(uinf), 4))

ρ(x, y) = 1.0 + 0.1 * exp(-30 * ((x - 0.5)^2 + (y - 0.5)^2))
u0 = initu(mesh, 4, [ρ,
                       (x, y) -> 0.3 * ρ(x, y),
                       (x, y) -> 0.05 * ρ(x, y),
                       (x, y) -> 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2) * ρ(x, y)])

# ---- correctness: Float64 GPU must match Float64 CPU to roundoff ----
ctx64 = DGContext(master, mesh)
phys_gpu = adapt(CuArray, phys)
r64_cpu = rinvexpl_ka(ctx64, phys, u0, 0.0)
r64_gpu = rinvexpl_ka(adapt(CuArray, ctx64), phys_gpu, CuArray(u0), 0.0)
rel64 = norm(Array(r64_gpu) .- r64_cpu) / norm(r64_cpu)
@printf "Float64 GPU vs CPU residual relative difference: %.3e\n" rel64
@assert rel64 < 1e-10 "GPU/CPU mismatch in Float64 — kernel bug"

# ---- Float32 (the precision actually used for performance): differences are
# rounding noise (atomic ordering, FMA contraction) amplified by inverse-mass
# conditioning, so only a loose check is meaningful ----
ctx_cpu = DGContext(master, mesh; T=Float32)
u32 = Float32.(u0)
r_cpu = rinvexpl_ka(ctx_cpu, phys, u32, 0.0f0)

ctx_gpu = adapt(CuArray, ctx_cpu)
u_gpu = CuArray(u32)
r_gpu = rinvexpl_ka(ctx_gpu, phys_gpu, u_gpu, 0.0f0)

rel32_gpu = norm(Float64.(Array(r_gpu)) .- r64_cpu) / norm(r64_cpu)
rel32_cpu = norm(Float64.(r_cpu) .- r64_cpu) / norm(r64_cpu)
@printf "Float32 error vs Float64 reference: CPU %.3e, GPU %.3e\n" rel32_cpu rel32_gpu
@assert rel32_gpu < 10 * max(rel32_cpu, 1e-6) "GPU Float32 error far exceeds CPU Float32 error"
@assert all(isfinite, Array(r_gpu))

# ---- timing: repeated residual evaluations with preallocated workspace ----
nrep = 50
ws_cpu = RinvWorkspace(ctx_cpu, 4)
rr_cpu = similar(u32)
rinvexpl!(rr_cpu, ctx_cpu, phys, u32, 0.0f0; ws=ws_cpu)  # warmup
t_cpu = @elapsed for _ in 1:nrep
    rinvexpl!(rr_cpu, ctx_cpu, phys, u32, 0.0f0; ws=ws_cpu)
end

ws_gpu = RinvWorkspace(ctx_gpu, 4)
rr_gpu = similar(u_gpu)
rinvexpl!(rr_gpu, ctx_gpu, phys_gpu, u_gpu, 0.0f0; ws=ws_gpu)  # warmup
t_gpu = @elapsed for _ in 1:nrep
    rinvexpl!(rr_gpu, ctx_gpu, phys_gpu, u_gpu, 0.0f0; ws=ws_gpu)
end

nt = size(mesh.t, 1)
@printf "mesh: %d elements, porder %d, %d residual evals\n" nt porder nrep
@printf "CPU (KA backend, %d threads): %.3f s   GPU: %.3f s   speedup: %.1fx\n" Threads.nthreads() t_cpu t_gpu t_cpu / t_gpu

# ---- a few RK4 steps on the GPU ----
uu = copy(u_gpu)
rk4_ka!(ctx_gpu, phys_gpu, uu, 0.0f0, 1f-4, 20; ws=ws_gpu)
@assert all(isfinite, Array(uu))
println("RK4 on GPU: OK")

# ============================ LDG viscous path ============================
println("\n-- LDG viscous path (convection-diffusion) --")

physv = DGPhysics(ConvectionDiffusionEquation(x -> SVector(-x[2], x[1]), 0.01);
                  boundary_conditions=(Dirichlet(0.0), Neumann(), Dirichlet(0.0), Neumann()),
                  stabilization=LDGStabilization(10.0, 0.5))
physv_gpu = adapt(CuArray, physv)
uv0 = initu(mesh, 1, [(x, y) -> exp(-4 * ((x - 0.5)^2 + (y - 0.5)^2))])

# Float64 GPU vs CPU parity (gradient and residual)
qv_cpu = getq_ka(ctx64, physv, uv0, 0.0)
qv_gpu = getq_ka(adapt(CuArray, ctx64), physv_gpu, CuArray(uv0), 0.0)
relq = norm(Array(qv_gpu) .- qv_cpu) / norm(qv_cpu)
rv_cpu = rldgexpl_ka(ctx64, physv, uv0, 0.0)
rv_gpu = rldgexpl_ka(adapt(CuArray, ctx64), physv_gpu, CuArray(uv0), 0.0)
relv = norm(Array(rv_gpu) .- rv_cpu) / norm(rv_cpu)
@printf "Float64 GPU vs CPU: getq %.3e, rldgexpl %.3e\n" relq relv
@assert relq < 1e-10 && relv < 1e-10 "LDG GPU/CPU mismatch in Float64 — kernel bug"

# Float32 timing, CPU KA backend vs GPU
uv32 = Float32.(uv0)
wsv_cpu = RldgWorkspace(ctx_cpu, 1)
rrv_cpu = similar(uv32)
rldgexpl!(rrv_cpu, ctx_cpu, physv, uv32, 0.0f0; ws=wsv_cpu)  # warmup
tv_cpu = @elapsed for _ in 1:nrep
    rldgexpl!(rrv_cpu, ctx_cpu, physv, uv32, 0.0f0; ws=wsv_cpu)
end

wsv_gpu = RldgWorkspace(ctx_gpu, 1)
uv_gpu = CuArray(uv32)
rrv_gpu = similar(uv_gpu)
rldgexpl!(rrv_gpu, ctx_gpu, physv_gpu, uv_gpu, 0.0f0; ws=wsv_gpu)  # warmup
tv_gpu = @elapsed for _ in 1:nrep
    rldgexpl!(rrv_gpu, ctx_gpu, physv_gpu, uv_gpu, 0.0f0; ws=wsv_gpu)
end
@printf "LDG residual, %d evals: CPU %.3f s   GPU %.3f s   speedup: %.1fx\n" nrep tv_cpu tv_gpu tv_cpu / tv_gpu

# a few viscous RK4 steps on the GPU
rk4_ka!(rldgexpl!, ctx_gpu, physv_gpu, uv_gpu, 0.0f0, 1f-4, 20; ws=wsv_gpu)
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

# ==================== CG matrix-free solver (Phase 4) ====================
println("\n-- CG matrix-free Krylov solver (Poisson) --")

cg_source(x, y) = 2π^2 * sin(π * x) * sin(π * y)
cg_param_p = (; κ=1.0, c=[0.0, 0.0], s=0.0)

nc_, pc_ = 65, 3
mesh_c = mkmesh_square(nc_, nc_, pc_, 0, 1)
master_c = Master(mesh_c, 4pc_)
@printf "CG mesh: %d elements, porder %d, %d CG nodes\n" size(mesh_c.t, 1) pc_ size(mesh_c.pcg, 1)

# Float64: direct reference, then CPU-backend and GPU iterations
uh_ref, energy_ref = cg_solve(mesh_c, master_c, cg_source, cg_param_p)
uh_cpu, e_cpu, it_cpu = cg_parsolve(mesh_c, master_c, cg_source, cg_param_p; tol=1e-12)
uh_gpu, e_gpu, it_gpu = cg_parsolve(mesh_c, master_c, cg_source, cg_param_p;
                                    tol=1e-12, ArrayT=CuArray)
rel_cpu = norm(uh_cpu .- uh_ref) / norm(uh_ref)
rel_gpu = norm(uh_gpu .- uh_ref) / norm(uh_ref)
@printf "Float64 vs direct: CPU %.3e (%d iters)   GPU %.3e (%d iters)\n" rel_cpu it_cpu rel_gpu it_gpu
@assert rel_cpu < 1e-8 && rel_gpu < 1e-8 "CG iterative/direct mismatch"
@assert isapprox(e_gpu, energy_ref; rtol=1e-8)

# Float32 on the GPU (same loose expectations as the other solvers)
uh32, e32, it32 = cg_parsolve(mesh_c, master_c, cg_source, cg_param_p;
                              T=Float32, tol=1e-6, ArrayT=CuArray)
rel32c = norm(Float64.(uh32) .- uh_ref) / norm(uh_ref)
@printf "Float32 GPU vs Float64 direct: %.3e (%d iters)\n" rel32c it32
@assert rel32c < 1e-3

# timing: whole solve (assembly is CPU either way; the iteration is the payload)
t_ccpu = @elapsed cg_parsolve(mesh_c, master_c, cg_source, cg_param_p; tol=1e-12)
t_cgpu = @elapsed cg_parsolve(mesh_c, master_c, cg_source, cg_param_p; tol=1e-12, ArrayT=CuArray)
t_cdir = @elapsed cg_solve(mesh_c, master_c, cg_source, cg_param_p)
@printf "CG solve: direct %.3f s   iterative CPU %.3f s   iterative GPU %.3f s (%.1fx vs CPU)\n" t_cdir t_ccpu t_cgpu t_ccpu / t_cgpu
println("CG on GPU: OK")

# ============ HDG Navier-Stokes + scalar transport (Phase 5) ============
println("\n-- HDG incompressible NS + temperature (heated cavity blocks) --")

Ra_, Pr_ = 1e4, 0.71
ν_ns = sqrt(Pr_ / Ra_)
κ_ns = 1 / sqrt(Ra_ * Pr_)
nn_, pn_ = 33, 3
mesh_n = mkmesh_square(nn_, nn_, pn_, 0, 1)
master_n = Master(mesh_n, 3 * (pn_ + 1))
npl_n, nt_n = size(mesh_n.dgnodes, 1), size(mesh_n.t, 1)
@printf "NS mesh: %d elements, porder %d, %d trace+pressure dofs\n" nt_n pn_ 2 * (pn_ + 1) * size(mesh_n.f, 1) + nt_n

dbc_ns(p) = [0.0, 0.0]
tbc_ns(p, tag) = tag == 4 ? (:d, 0.5) : tag == 2 ? (:d, -0.5) : (:n, 0.0)
θ0_ns = [0.5 - mesh_n.dgnodes[k, 1, it] for k in 1:npl_n, it in 1:nt_n]
src_ns = zeros(npl_n, 2, nt_n)
src_ns[:, 2, :] .= θ0_ns
dtinv_ns = 4.0

# legacy CPU reference (one Newton step / one transport step)
t_leg = @elapsed ref = hdg_ns_step(master_n, mesh_n, ν_ns, dbc_ns;
                                   τ=1.0, source=src_ns, dtinv=dtinv_ns)
t_legcd = @elapsed refθ = hdg_cd_step(master_n, mesh_n, κ_ns, tbc_ns;
                                      τ=1.0, u=ref.u, Λ=ref.Λ, θold=θ0_ns, dtinv=dtinv_ns)

for (label, AT) in (("CPU backend", Array), ("GPU", CuArray))
    b1 = hdg_ns_step_batched(master_n, mesh_n, ν_ns, dbc_ns;
                             τ=1.0, source=src_ns, dtinv=dtinv_ns, ArrayT=AT)
    ns_rel_u = norm(b1.u .- ref.u) / norm(ref.u)
    ns_rel_Λ = norm(b1.Λ .- ref.Λ) / norm(ref.Λ)
    # warm second Newton step reuses pattern + numeric refactorization
    t_ns = @elapsed b2 = hdg_ns_step_batched(master_n, mesh_n, ν_ns, dbc_ns;
                                             τ=1.0, source=src_ns, u=b1.u, Λ=b1.Λ,
                                             uold=b1.u, dtinv=dtinv_ns, cache=b1.cache)
    c1 = hdg_cd_step_batched(master_n, mesh_n, κ_ns, tbc_ns;
                             τ=1.0, u=b1.u, Λ=b1.Λ, θold=θ0_ns, dtinv=dtinv_ns, ArrayT=AT)
    ns_rel_θ = norm(c1.θ .- refθ.θ) / norm(refθ.θ)
    t_cd = @elapsed hdg_cd_step_batched(master_n, mesh_n, κ_ns, tbc_ns;
                                        τ=1.0, u=b1.u, Λ=b1.Λ, θold=θ0_ns,
                                        dtinv=dtinv_ns, cache=c1.cache)
    @printf "%s: NS parity u %.3e Λ %.3e, θ parity %.3e | NS step %.2f s (legacy %.2f s), θ step %.2f s (legacy %.2f s)\n" label ns_rel_u ns_rel_Λ ns_rel_θ t_ns t_leg t_cd t_legcd
    @assert ns_rel_u < 1e-7 && ns_rel_Λ < 1e-7 && ns_rel_θ < 1e-10
end
println("HDG NS/CD batched on GPU: OK")
