# CUDA validation pass for 3D continuous Galerkin: the matrix-free
# Jacobi-preconditioned Krylov path (`cg_parsolve` with ArrayT=CuArray) on a
# tetrahedral box, checked against the sparse-direct CPU solution, plus the
# user-facing CGProblem/ConjugateGradient route.
#
# CUDA.jl lives in the shared "@cuda" environment (see examples/dg/run_ka_cuda.jl):
#     JULIA_LOAD_PATH='@;@cuda;@v#.#;@stdlib' julia +1.12 --project=. examples/cg/run3d_cg_cuda.jl

using TwoDG
using CUDA
using LinearAlgebra
using Printf

@assert CUDA.functional() "CUDA is not functional on this machine"
println("GPU: ", CUDA.name(CUDA.device()))

exact(x, y, z) = sin(π * x) * sin(π * y) * sin(π * z)
source(x, y, z) = 3π^2 * exact(x, y, z)
param_p = (; κ=1.0, c=[0.0, 0.0, 0.0], s=0.0)       # Poisson → Krylov cg
param_cd = (; κ=0.7, c=[1.5, -0.4, 0.8], s=0.3)     # convection → Krylov gmres

porder = 3
n = 9
mesh = mkmesh_box(n, n, n, porder)
master = ReferenceElement(mesh, 4porder)
@printf "mesh: %d tets, porder %d, %d CG nodes\n" size(mesh.t, 1) porder size(mesh.pcg, 1)

# Float64: sparse-direct reference, then CPU-backend and GPU Krylov
uh_ref, energy_ref = cg_solve(mesh, master, source, param_p)
uh_cpu, e_cpu, it_cpu = cg_parsolve(mesh, master, source, param_p; tol=1e-12)
uh_gpu, e_gpu, it_gpu = cg_parsolve(mesh, master, source, param_p;
                                    tol=1e-12, ArrayT=CuArray)
rel_cpu = norm(uh_cpu .- uh_ref) / norm(uh_ref)
rel_gpu = norm(uh_gpu .- uh_ref) / norm(uh_ref)
@printf "Poisson F64 vs direct: CPU %.3e (%d iters)   GPU %.3e (%d iters)\n" rel_cpu it_cpu rel_gpu it_gpu
@assert rel_cpu < 1e-8 && rel_gpu < 1e-8 "CG iterative/direct mismatch"
@assert isapprox(e_gpu, energy_ref; rtol=1e-8)
@printf "L2 error vs exact: %.3e\n" l2error(mesh, uh_gpu, exact)

# convection-diffusion exercises the gmres branch on the device
uh_ref2, _ = cg_solve(mesh, master, source, param_cd)
uh_gpu2, _, it2 = cg_parsolve(mesh, master, source, param_cd; tol=1e-12, ArrayT=CuArray)
rel2 = norm(uh_gpu2 .- uh_ref2) / norm(uh_ref2)
@printf "CD F64 GPU gmres vs direct: %.3e (%d iters)\n" rel2 it2
@assert rel2 < 1e-7

# Float32 on the GPU (loose: single-precision Krylov on a p=3 stiffness matrix)
uh32, _, it32 = cg_parsolve(mesh, master, source, param_p;
                            T=Float32, tol=1e-6, ArrayT=CuArray)
rel32 = norm(Float64.(uh32) .- uh_ref) / norm(uh_ref)
@printf "Poisson F32 GPU vs F64 direct: %.3e (%d iters)\n" rel32 it32
@assert rel32 < 1e-3

# timing: whole solve (assembly is CPU either way; the iteration is the payload)
t_dir = @elapsed cg_solve(mesh, master, source, param_p)
t_cpu = @elapsed cg_parsolve(mesh, master, source, param_p; tol=1e-12)
t_gpu = @elapsed cg_parsolve(mesh, master, source, param_p; tol=1e-12, ArrayT=CuArray)
@printf "solve: direct %.2f s   iterative CPU %.2f s   iterative GPU %.2f s (%.1fx vs CPU)\n" t_dir t_cpu t_gpu t_cpu / t_gpu

# user-facing route
prob = CGProblem(PoissonEquation{3}(), mesh; source)
sol = solve(prob, ConjugateGradient())
@printf "CGProblem + ConjugateGradient: %d iters, L2 error %.3e\n" sol.iterations l2error(sol, exact)
@assert sol.iterations > 0

println("3D CG CUDA pass complete.")
