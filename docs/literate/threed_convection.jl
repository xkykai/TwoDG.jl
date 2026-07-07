# # 3D transport on a tetrahedral box
#
# Everything after mesh construction works exactly as in 2D: the same
# equations, boundary conditions, `Problem` types, and `solve` calls are
# dimension-generic, with the dimension inferred from the mesh. This
# entry-level tutorial advects a Gaussian bump through the unit cube with
# the discontinuous Galerkin method and verifies the design convergence
# rate ``p+1``.
#
# The constant-velocity transport equation
# ``u_t + \nabla\cdot(\mathbf{v} u) = 0`` translates the initial profile
# unchanged: ``u(\mathbf{x}, t) = u_0(\mathbf{x} - \mathbf{v} t)`` — an
# exact solution to measure against.

using TwoDG
using StaticArrays
using Printf

v = SVector(1.0, 0.5, 0.25)
u0(x, y, z) = exp(-30 * ((x - 0.4)^2 + (y - 0.45)^2 + (z - 0.4)^2))

# ## Mesh and problem
#
# [`mkmesh_box`](@ref) tiles a structured grid of cubes into 6 tetrahedra
# each (the Kuhn split — conforming and uniformly refinable). The six box
# faces carry the boundary names `:left/:right/:bottom/:top/:front/:back`;
# here every boundary gets the same far-field condition, so a plain tuple
# entry per tag works too.

porder = 2
mesh = mkmesh_box(5, 5, 5, porder)
size(mesh.t)

# A scalar convection equation infers its dimension from the velocity
# vector. `compute_dt` gives a CFL-limited step (inscribed-sphere diameter):

eq = ConvectionEquation(v)
prob = DGProblem(eq, mesh; bc = ntuple(_ -> FarField(SVector(0.0)), 6),
                 u0 = [u0])
dt = compute_dt(prob; cfl = 0.3)

# ## Solve and measure

tfinal = 0.2
nstep = ceil(Int, tfinal / dt)
sol = solve(prob, RK4(); dt = tfinal / nstep, tfinal)
exact(x, y, z) = u0(x - v[1] * tfinal, y - v[2] * tfinal, z - v[3] * tfinal)
l2error(mesh, sol.u[:, 1, :], exact)

# ## Convergence
#
# Halving ``h`` should cut the error by ``2^{p+1}``. (The wave must cross
# a meaningful fraction of the domain — measuring at tiny final times shows
# the truncation order ``p`` instead of the solution order ``p+1``.)

errs = map((3, 5)) do n
    m = mkmesh_box(n, n, n, porder)
    pr = DGProblem(eq, m; bc = ntuple(_ -> FarField(SVector(0.0)), 6), u0 = [u0])
    s = solve(pr, RK4(); dt = tfinal / nstep, tfinal)
    l2error(m, s.u[:, 1, :], exact)
end
@printf "h = 1/2: %.3e   h = 1/4: %.3e   rate %.2f (design %d)\n" errs[1] errs[2] log2(errs[1] / errs[2]) porder + 1

# ## ParaView output
#
# 3D fields are best inspected in ParaView. With WriteVTK.jl loaded,
# [`save_vtk`](@ref) writes high-order Lagrange cells that render the
# curved polynomial solution exactly:
#
# ```julia
# using WriteVTK
# save_vtk(mesh, sol.u, "convection"; names = (:u,))
# ```
