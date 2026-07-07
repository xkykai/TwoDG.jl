# # 3D compressible Euler and the GPU
#
# The isentropic vortex is the classic exact-solution benchmark for
# compressible flow solvers: a vortex in isentropic balance advects with
# the free stream unchanged. We run it on tetrahedra with a vortex axis
# aligned with ``z`` and a free stream with all three components nonzero,
# so every flux direction and boundary is exercised — then show how the
# same problem moves to a GPU.

using TwoDG
using StaticArrays
using Printf

γ = 1.4
β, R = 1.0, 0.25                    # vortex strength and core radius
v∞ = SVector(1.0, 0.5, 0.25)

function conserved(x, y, z, t)
    x̂ = (x - 0.5 - v∞[1] * t) / R
    ŷ = (y - 0.5 - v∞[2] * t) / R
    f = exp((1 - x̂^2 - ŷ^2) / 2)
    Θ = 1 - (γ - 1) * β^2 / (8γ * π^2) * f^2            # temperature
    ρ = Θ^(1 / (γ - 1))
    vel = SVector(v∞[1] - β / (2π) * ŷ * f, v∞[2] + β / (2π) * x̂ * f, v∞[3])
    return SVector(ρ, ρ * vel[1], ρ * vel[2], ρ * vel[3],
                   Θ^(γ / (γ - 1)) / (γ - 1) + ρ * sum(abs2, vel) / 2)
end
uex(x, t) = conserved(x[1], x[2], x[3], t);

# ## Problem setup
#
# `EulerEquations{3}` has five conserved variables ``(\rho, \rho u, \rho v,
# \rho w, \rho E)`` and the standard Roe flux. Time-dependent Dirichlet
# data supplies the exact vortex on all six boundaries.

porder = 2
mesh = mkmesh_box(5, 5, 5, porder)
prob = DGProblem(EulerEquations{3}(γ = γ), mesh;
                 bc = ntuple(_ -> Dirichlet(uex), 6),
                 u0 = [(x, y, z) -> conserved(x, y, z, 0.0)[c] for c in 1:5])

tfinal = 0.1
dt = compute_dt(prob; cfl = 0.3)
nstep = ceil(Int, tfinal / dt)
sol = solve(prob, RK4(); dt = tfinal / nstep, tfinal)
err = l2error(mesh, sol.u[:, 1, :], (x, y, z) -> conserved(x, y, z, tfinal)[1])
@printf "%d tets, dt = %.2e: L2(ρ) error %.3e\n" size(mesh.t, 1) dt err

# (Refining `5 → 9` reduces this at the design rate ``p+1``; the full study
# lives in `examples/dg3d/run3d_euler_vortex.jl`.)
#
# ## Moving to the GPU
#
# The residual path is written once in KernelAbstractions.jl and runs on
# any device: move the geometry cache, physics, and state over with
# `Adapt.adapt` and call the same functions. In `Float32` on a laptop-class
# RTX 3050 Ti, the 3D Euler residual runs ~3.4× faster than the 8-thread
# CPU backend at 6000 tets (and the gap grows with mesh size — small
# meshes are launch-overhead-bound):
#
# ```julia
# using CUDA, Adapt
#
# ctx  = DGContext(ReferenceElement(mesh), mesh; T = Float32)   # Float32 tables
# phys = DGPhysics(EulerEquations{3}(γ = 1.4f0);
#                  boundary_conditions = ntuple(_ -> Dirichlet(uex), 6))
#
# ctx_gpu  = adapt(CuArray, ctx)
# phys_gpu = adapt(CuArray, phys)
# u_gpu    = CuArray(Float32.(initu(mesh, 5, ics)))
#
# ws = RinvWorkspace(ctx_gpu, 5)                 # staging buffers on the device
# rk4_ka!(ctx_gpu, phys_gpu, u_gpu, 0.0f0, dt, nstep; ws)
# ```
#
# Or, through the high-level interface, pass `ArrayT = CuArray` to `solve`:
#
# ```julia
# sol = solve(prob, RK4(); dt, tfinal, ArrayT = CuArray, T = Float32)
# ```
#
# Two practical notes for consumer GPUs:
#
# - **Use `Float32`** for explicit DG — consumer cards have weak `Float64`
#   throughput. The `Float32` residual differs from `Float64` by ~``10^{-3}``
#   relative at ``p = 3`` (inverse-mass conditioning; identical on CPU and
#   GPU).
# - **HDG is the exception**: its GMRES trace solve is memory-bound, so
#   `Float64` costs little there and single precision can stagnate before
#   tight tolerances.
