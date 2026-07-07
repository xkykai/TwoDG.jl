# CUDA validation pass for the 3D HDG batched paths: Poisson (local solves +
# GMRES trace solve), incompressible Navier-Stokes (batched Newton step with
# cache reuse), and scalar transport — GPU vs CPU-backend parity and timings.
# The trace saddle solve stays a CPU sparse LU by design; the GPU payload is
# the batched per-element assembly, local solves, and recovery.
#
# CUDA.jl lives in the shared "@cuda" environment (see examples/dg/run_ka_cuda.jl):
#     JULIA_LOAD_PATH='@;@cuda;@v#.#;@stdlib' julia +1.12 --project=. examples/hdg3d/runhdg3d_ka_cuda.jl

using TwoDG
using CUDA
using Adapt
using LinearAlgebra
using Printf

@assert CUDA.functional() "CUDA is not functional on this machine"
println("GPU: ", CUDA.name(CUDA.device()))

# ============================ Poisson on the box ============================
println("\n-- HDG 3D Poisson: batched local solves + GMRES trace solve --")

exact(x, y, z) = sin(π * x) * sin(π * y) * sin(π * z)
source(p) = reshape(3π^2 .* sin.(π .* p[:, 1]) .* sin.(π .* p[:, 2]) .*
                    sin.(π .* p[:, 3]), :, 1)
dbc(p) = zeros(size(p, 1), 1)
param = Dict(:kappa => 1.0, :c => zeros(3), :taud => 1.0)

porder = 2
n = 9
mesh = mkmesh_box(n, n, n, porder)
master = ReferenceElement(mesh, 4 * (porder + 1))
nt = size(mesh.t, 1)
@printf "mesh: %d tets, porder %d, %d faces\n" nt porder size(mesh.f, 1)

# batched local solves: GPU vs CPU backend to roundoff
print("HDGBatch build (CPU geometry): ")
t_bb = @elapsed batch = HDGBatch(master, mesh, source, param)
@printf "%.2f s\n" t_bb
loc_cpu = hdg_local_solves(batch)
batch_gpu = adapt(CuArray, batch)
loc_gpu = hdg_local_solves(batch_gpu)  # warmup/compile
rel_ae = norm(Array(loc_gpu.ae) .- loc_cpu.ae) / norm(loc_cpu.ae)
rel_fe = norm(Array(loc_gpu.fe) .- loc_cpu.fe) / norm(loc_cpu.fe)
@printf "batched local solves, GPU vs CPU backend: ae %.3e, fe %.3e\n" rel_ae rel_fe
@assert rel_ae < 1e-10 && rel_fe < 1e-10

t_lcpu = @elapsed hdg_local_solves(batch)
CUDA.@sync (loc2 = hdg_local_solves(batch_gpu))
t_lgpu = @elapsed CUDA.@sync hdg_local_solves(batch_gpu)
@printf "local solves: CPU %.3f s   GPU %.3f s (%.1fx)\n" t_lcpu t_lgpu t_lcpu / t_lgpu

# Float32 local solves on the GPU: loose check against the Float64 reference
batch32_gpu = adapt(CuArray, HDGBatch(master, mesh, source, param; T=Float32))
loc32 = hdg_local_solves(batch32_gpu)
rel32 = norm(Float64.(Array(loc32.ae)) .- loc_cpu.ae) / norm(loc_cpu.ae)
@printf "Float32 GPU local solves vs Float64: %.3e\n" rel32
@assert rel32 < 1e-3

# end-to-end: direct CPU reference vs GPU-batched GMRES driver
ud, qd, uhd = hdg_direct_batched(master, mesh, source, dbc, param)
t_dir = @elapsed hdg_direct_batched(master, mesh, source, dbc, param)
u_gpu, q_gpu, uh_gpu, niter = hdg_parsolve_batched(master, mesh, source, dbc, param;
                                                   ArrayT=CuArray, tol=1e-12, restart=200)
t_gmres = @elapsed hdg_parsolve_batched(master, mesh, source, dbc, param;
                                        ArrayT=CuArray, tol=1e-12, restart=200)
rel_u = norm(vec(u_gpu) .- vec(ud)) / norm(vec(ud))
rel_q = norm(vec(q_gpu) .- vec(qd)) / norm(vec(qd))
@printf "GPU GMRES (%d iters) vs CPU direct: u %.3e, q %.3e\n" niter rel_u rel_q
@assert rel_u < 1e-8 && rel_q < 1e-8
@printf "end-to-end: direct CPU %.2f s   GPU GMRES %.2f s\n" t_dir t_gmres
err_u = l2error(mesh, u_gpu[:, 1, :], exact)
@printf "L2 error vs exact: %.3e\n" err_u
println("HDG 3D Poisson on GPU: OK")

# ================= Navier-Stokes: Beltrami flow on the box =================
println("\n-- HDG 3D NS: batched Newton step (Beltrami) --")

a, d = π / 4, π / 2
ν = 1.0
uB(x, y, z) = [-a * (exp(a * x) * sin(a * y + d * z) + exp(a * z) * cos(a * x + d * y)),
               -a * (exp(a * y) * sin(a * z + d * x) + exp(a * x) * cos(a * y + d * z)),
               -a * (exp(a * z) * sin(a * x + d * y) + exp(a * y) * cos(a * z + d * x))]
fB(p) = ν * d^2 .* uB(p[1], p[2], p[3])
dbcB(p) = uB(p[1], p[2], p[3])
τ = 2.0

nn = 5
mesh_n = mkmesh_box(nn, nn, nn, porder)
master_n = ReferenceElement(mesh_n, 3 * (porder + 1))
nt_n = size(mesh_n.t, 1)
nps = size(mesh_n.elcon, 1) ÷ 4
@printf "NS mesh: %d tets, %d trace+pressure dofs\n" nt_n 3 * nps * size(mesh_n.f, 1) + nt_n

# cold Newton step: CPU backend vs GPU parity
s_cpu = hdg_ns_step_batched(master_n, mesh_n, ν, dbcB; τ, source=fB)
s_gpu = hdg_ns_step_batched(master_n, mesh_n, ν, dbcB; τ, source=fB, ArrayT=CuArray)
rel_nsu = norm(s_gpu.u .- s_cpu.u) / norm(s_cpu.u)
rel_nsΛ = norm(s_gpu.Λ .- s_cpu.Λ) / norm(s_cpu.Λ)
rel_nsg = norm(s_gpu.gradu .- s_cpu.gradu) / norm(s_cpu.gradu)
@printf "cold NS step, GPU vs CPU backend: u %.3e, Λ %.3e, ∇u %.3e\n" rel_nsu rel_nsΛ rel_nsg
@assert rel_nsu < 1e-8 && rel_nsΛ < 1e-8

# warm cached step (pattern + numeric refactorization reuse) — the time-loop shape
t_ns_cpu = @elapsed hdg_ns_step_batched(master_n, mesh_n, ν, dbcB; τ, source=fB,
                                        u=s_cpu.u, Λ=s_cpu.Λ, cache=s_cpu.cache)
t_ns_gpu = @elapsed hdg_ns_step_batched(master_n, mesh_n, ν, dbcB; τ, source=fB,
                                        u=s_gpu.u, Λ=s_gpu.Λ, cache=s_gpu.cache)
@printf "warm Newton step: CPU %.2f s   GPU %.2f s\n" t_ns_cpu t_ns_gpu

# full Newton iteration on the GPU: must converge to the exact Beltrami field
u = Λ = cache = nothing
res = nothing
for iter in 1:10
    global res = hdg_ns_step_batched(master_n, mesh_n, ν, dbcB; τ, source=fB,
                                     u, Λ, cache, ArrayT=CuArray)
    Δ = Λ === nothing ? Inf : norm(res.Λ .- Λ) / max(norm(res.Λ), eps())
    @printf "  Newton %d: Δλ = %.2e\n" iter Δ
    global u, Λ, cache = res.u, res.Λ, res.cache
    Δ < 1e-10 && break
end
err_B = sqrt(sum(abs2, [l2error(mesh_n, res.u[:, c, :],
                                (x, y, z) -> uB(x, y, z)[c]) for c in 1:3]))
divmax = maximum(abs.(res.gradu[:, 1, :] .+ res.gradu[:, 5, :] .+ res.gradu[:, 9, :]))
@printf "GPU Beltrami: |u - u_B| = %.3e, max |div u| = %.2e\n" err_B divmax
@assert err_B < 0.05 && divmax < 1e-5
println("HDG 3D NS on GPU: OK")

# ==================== scalar transport with the NS field ====================
println("\n-- HDG 3D scalar transport (batched CD step) --")

κ = 0.05
tbc(p, tag) = tag == 1 ? (:d, 0.5) : tag == 2 ? (:d, -0.5) : (:n, 0.0)
θold = 0.5 .- mesh_n.dgnodes[:, 1, :] ./ 2
c_cpu = hdg_cd_step_batched(master_n, mesh_n, κ, tbc; τ=1.0, u=s_cpu.u, Λ=s_cpu.Λ,
                            θold, dtinv=4.0)
c_gpu = hdg_cd_step_batched(master_n, mesh_n, κ, tbc; τ=1.0, u=s_cpu.u, Λ=s_cpu.Λ,
                            θold, dtinv=4.0, ArrayT=CuArray)
rel_θ = norm(c_gpu.θ .- c_cpu.θ) / norm(c_cpu.θ)
rel_qθ = norm(c_gpu.q .- c_cpu.q) / norm(c_cpu.q)
@printf "CD step, GPU vs CPU backend: θ %.3e, q %.3e\n" rel_θ rel_qθ
@assert rel_θ < 1e-10 && rel_qθ < 1e-9
t_cd_cpu = @elapsed hdg_cd_step_batched(master_n, mesh_n, κ, tbc; τ=1.0, u=s_cpu.u,
                                        Λ=s_cpu.Λ, θold, dtinv=4.0, cache=c_cpu.cache)
t_cd_gpu = @elapsed hdg_cd_step_batched(master_n, mesh_n, κ, tbc; τ=1.0, u=s_cpu.u,
                                        Λ=s_cpu.Λ, θold, dtinv=4.0, cache=c_gpu.cache)
@printf "warm CD step: CPU %.2f s   GPU %.2f s\n" t_cd_cpu t_cd_gpu
println("HDG 3D transport on GPU: OK")

println("\n3D HDG CUDA pass complete.")
