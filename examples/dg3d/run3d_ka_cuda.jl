# CUDA validation pass for the 3D DG/LDG residual path: Float64 GPU/CPU
# parity to roundoff, Float32 sanity, and residual-evaluation timings on a
# structured tet box.
#
# CUDA.jl is deliberately NOT a dependency of TwoDG; install it once into the
# shared "@cuda" environment so it's picked up via environment stacking:
#     julia +1.12 -e 'using Pkg; Pkg.activate("cuda"; shared=true); Pkg.add("CUDA")'
# then run (JULIA_LOAD_PATH separator is ';' on Windows, ':' elsewhere):
#     JULIA_LOAD_PATH='@;@cuda;@v#.#;@stdlib' julia +1.12 --project=. examples/dg3d/run3d_ka_cuda.jl

using TwoDG
using CUDA
using Adapt
using StaticArrays
using LinearAlgebra
using Printf

@assert CUDA.functional() "CUDA is not functional on this machine"
println("GPU: ", CUDA.name(CUDA.device()))

γ = 1.4
v∞ = SVector(1.0, 0.5, 0.25)
u∞ = [1.0, v∞[1], v∞[2], v∞[3], 1.0 / (γ - 1) + 0.5 * sum(abs2, v∞)]

# smooth, strictly physical Euler state: density bump advected by the free stream
ρ3(x, y, z) = 1.0 + 0.1 * exp(-30 * ((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2))
euler_ic3 = [ρ3,
             (x, y, z) -> v∞[1] * ρ3(x, y, z),
             (x, y, z) -> v∞[2] * ρ3(x, y, z),
             (x, y, z) -> v∞[3] * ρ3(x, y, z),
             (x, y, z) -> 1.0 / (γ - 1) + 0.5 * sum(abs2, v∞) * ρ3(x, y, z)]

# ---- Float64 parity mesh (small: dense curved-layout tables are heavy in 3D) ----
porder = 3
mesh64 = mkmesh_box(6, 6, 6, porder)
master64 = ReferenceElement(mesh64)
phys = DGPhysics(EulerEquations{3}(γ=γ); boundary_conditions=ntuple(_ -> FarField(u∞), 6))
phys_gpu = adapt(CuArray, phys)
u64 = initu(mesh64, 5, euler_ic3)

ctx64 = DGContext(master64, mesh64)
r64_cpu = rinvexpl_ka(ctx64, phys, u64, 0.0)
r64_gpu = rinvexpl_ka(adapt(CuArray, ctx64), phys_gpu, CuArray(u64), 0.0)
rel64 = norm(Array(r64_gpu) .- r64_cpu) / norm(r64_cpu)
@printf "Euler 3D Float64 GPU vs CPU residual relative difference: %.3e\n" rel64
@assert rel64 < 1e-10 "GPU/CPU mismatch in Float64 — kernel bug"

# ---- timing mesh, Float32 (the performance configuration on consumer GPUs) ----
n_t = 11
mesh = mkmesh_box(n_t, n_t, n_t, porder)
master = ReferenceElement(mesh)
nt = size(mesh.t, 1)
u0 = initu(mesh, 5, euler_ic3)

ctx_cpu = DGContext(master, mesh; T=Float32)
u32 = Float32.(u0)
r_cpu = rinvexpl_ka(ctx_cpu, phys, u32, 0.0f0)

ctx_gpu = adapt(CuArray, ctx_cpu)
u_gpu = CuArray(u32)
r_gpu = rinvexpl_ka(ctx_gpu, phys_gpu, u_gpu, 0.0f0)

# Float32 differences are rounding noise (atomics, FMA) amplified by the
# inverse-mass conditioning; compare both backends against a Float64 reference
r64_ref = rinvexpl_ka(DGContext(master, mesh), phys, u0, 0.0)
rel32_cpu = norm(Float64.(r_cpu) .- r64_ref) / norm(r64_ref)
rel32_gpu = norm(Float64.(Array(r_gpu)) .- r64_ref) / norm(r64_ref)
@printf "Euler 3D Float32 error vs Float64 reference: CPU %.3e, GPU %.3e\n" rel32_cpu rel32_gpu
@assert rel32_gpu < 10 * max(rel32_cpu, 1e-6) "GPU Float32 error far exceeds CPU Float32 error"
@assert all(isfinite, Array(r_gpu))

nrep = 50
ws_cpu = RinvWorkspace(ctx_cpu, 5)
rr_cpu = similar(u32)
rinvexpl!(rr_cpu, ctx_cpu, phys, u32, 0.0f0; ws=ws_cpu)  # warmup
t_cpu = @elapsed for _ in 1:nrep
    rinvexpl!(rr_cpu, ctx_cpu, phys, u32, 0.0f0; ws=ws_cpu)
end

ws_gpu = RinvWorkspace(ctx_gpu, 5)
rr_gpu = similar(u_gpu)
rinvexpl!(rr_gpu, ctx_gpu, phys_gpu, u_gpu, 0.0f0; ws=ws_gpu)  # warmup
CUDA.@sync rinvexpl!(rr_gpu, ctx_gpu, phys_gpu, u_gpu, 0.0f0; ws=ws_gpu)
t_gpu = @elapsed CUDA.@sync for _ in 1:nrep
    rinvexpl!(rr_gpu, ctx_gpu, phys_gpu, u_gpu, 0.0f0; ws=ws_gpu)
end

@printf "Euler 3D mesh: %d tets, porder %d, %d residual evals (Float32)\n" nt porder nrep
@printf "CPU (KA backend, %d threads): %.3f s   GPU: %.3f s   speedup: %.1fx\n" Threads.nthreads() t_cpu t_gpu t_cpu / t_gpu

# a few RK4 steps on the GPU
uu = copy(u_gpu)
rk4_ka!(ctx_gpu, phys_gpu, uu, 0.0f0, 1f-4, 20; ws=ws_gpu)
@assert all(isfinite, Array(uu))
println("Euler 3D RK4 on GPU: OK")

# ============================ LDG viscous path ============================
println("\n-- LDG viscous path (3D convection-diffusion) --")

physv = DGPhysics(ConvectionDiffusionEquation(SVector(1.0, 0.5, 0.25), 0.01);
                  boundary_conditions=ntuple(_ -> Dirichlet(0.0), 6),
                  stabilization=LDGStabilization(10.0, 0.0))
physv_gpu = adapt(CuArray, physv)
bump3(x, y, z) = exp(-30 * ((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2))

# Float64 GPU vs CPU parity (gradient and residual) on the small mesh
uv64 = initu(mesh64, 1, [bump3])
qv_cpu = getq_ka(ctx64, physv, uv64, 0.0)
qv_gpu = getq_ka(adapt(CuArray, ctx64), physv_gpu, CuArray(uv64), 0.0)
relq = norm(Array(qv_gpu) .- qv_cpu) / norm(qv_cpu)
rv_cpu = rldgexpl_ka(ctx64, physv, uv64, 0.0)
rv_gpu = rldgexpl_ka(adapt(CuArray, ctx64), physv_gpu, CuArray(uv64), 0.0)
relv = norm(Array(rv_gpu) .- rv_cpu) / norm(rv_cpu)
@printf "LDG 3D Float64 GPU vs CPU: getq %.3e, rldgexpl %.3e\n" relq relv
@assert relq < 1e-10 && relv < 1e-10 "LDG GPU/CPU mismatch in Float64 — kernel bug"

# Float32 timing on the big mesh
uv32 = Float32.(initu(mesh, 1, [bump3]))
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
CUDA.@sync rldgexpl!(rrv_gpu, ctx_gpu, physv_gpu, uv_gpu, 0.0f0; ws=wsv_gpu)
tv_gpu = @elapsed CUDA.@sync for _ in 1:nrep
    rldgexpl!(rrv_gpu, ctx_gpu, physv_gpu, uv_gpu, 0.0f0; ws=wsv_gpu)
end
@printf "LDG 3D residual, %d evals (Float32): CPU %.3f s   GPU %.3f s   speedup: %.1fx\n" nrep tv_cpu tv_gpu tv_cpu / tv_gpu

# a few viscous RK4 steps on the GPU
rk4_ka!(rldgexpl!, ctx_gpu, physv_gpu, uv_gpu, 0.0f0, 1f-4, 20; ws=wsv_gpu)
@assert all(isfinite, Array(uv_gpu))
println("LDG 3D RK4 on GPU: OK")

println("\n3D DG CUDA pass complete.")
