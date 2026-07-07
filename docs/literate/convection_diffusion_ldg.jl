# # Convection–diffusion with LDG
#
# This tutorial solves the scalar convection–diffusion equation
#
# ```math
# \partial_t u + \nabla \cdot (\boldsymbol{v} u) = \kappa \Delta u
# ```
#
# with the explicit **local discontinuous Galerkin** (LDG) method
# (Cockburn & Shu, SINUM 1998): the gradient is reconstructed as an
# auxiliary DG variable with one-sided ("alternating") face traces, so the
# viscous term keeps the design order `p + 1` without a globally coupled
# solve. As always, the claim is verified with a convergence study.

using TwoDG
using StaticArrays
using Printf

# ## The heat equation as a verification case
#
# With ``\boldsymbol{v} = 0``, ``\kappa = 0.05``, homogeneous Dirichlet
# boundaries, and ``u_0 = \sin(\pi x)\sin(\pi y)`` on the unit square, the
# exact solution is pure exponential decay,
# ``u(x, y, t) = u_0(x, y)\, e^{-2\pi^2 \kappa t}``.

κ  = 0.05
u0(x, y) = sin(π * x) * sin(π * y)
exact(x, y, t) = u0(x, y) * exp(-2π^2 * κ * t);

# A [`DGProblem`](@ref) bundles the equation, the boundary conditions (one
# typed object per named boundary), and the initial condition. Diffusive
# equations additionally carry an [`LDGStabilization`](@ref) — the `c11`
# penalty on the primal face jumps (`10.0` on the boundary, `0.0` in the
# interior is a robust default; the equation's `default_stabilization`
# applies when you omit it):

mesh = mkmesh_square(9, 9, 2, 0, 1)
eq   = ConvectionDiffusionEquation(SVector(0.0, 0.0), κ)
bc   = (bottom = Dirichlet(0.0), right = Dirichlet(0.0),
        top    = Dirichlet(0.0), left  = Dirichlet(0.0))
prob = DGProblem(eq, mesh; bc, u0 = [u0],
                 stabilization = LDGStabilization(10.0, 0.0));

# Explicit diffusion pays a quadratic CFL price, ``\Delta t \lesssim
# h^2 / (\kappa\,(2p+1)^2)`` — and the LDG penalty terms tighten the
# stability constant beyond that estimate, so we step well inside the limit
# and scale ``\Delta t \propto h^2`` under refinement. The step divides
# `tfinal` exactly (the decay factor is time-sensitive):

tfinal = 0.1
sol    = solve(prob, RK4(); dt = 2e-4, tfinal)
l2error(sol, (x, y) -> exact(x, y, tfinal))

# ## Convergence
#
# Refine at fixed `p = 2` and check the `p + 1` rate. The time error of RK4
# at these steps scales like ``\Delta t^4 \sim h^8`` — far below the
# ``h^{p+1}`` spatial error, so the measured rate is purely spatial.

function heat_error(m, porder)
    mesh = mkmesh_square(m, m, porder, 0, 1)
    prob = DGProblem(ConvectionDiffusionEquation(SVector(0.0, 0.0), κ), mesh;
                     bc, u0 = [u0], stabilization = LDGStabilization(10.0, 0.0))
    dt = 2e-4 * (8 / (m - 1))^2          # diffusive limit: dt ∝ h²
    sol = solve(prob, RK4(); dt, tfinal)
    return l2error(sol, (x, y) -> exact(x, y, tfinal))
end

ms = [5, 9, 17]
errs = [heat_error(m, 2) for m in ms]
@printf "p = 2      h        ‖u - uₕ‖      rate\n"
for i in eachindex(ms)
    h = 1 / (ms[i] - 1)
    if i == 1
        @printf "      %8.4f   %11.3e      --\n" h errs[i]
    else
        @printf "      %8.4f   %11.3e   %5.2f\n" h errs[i] log2(errs[i-1] / errs[i])
    end
end

# ## Adding convection
#
# The convective velocity can be a constant vector or a function of
# position (explicit DG solvers only) — here a solid-body rotation about
# the domain center transporting the decaying profile:

eq_rot   = ConvectionDiffusionEquation(x -> SVector(0.5 - x[2], x[1] - 0.5), 0.005)
prob_rot = DGProblem(eq_rot, mesh; bc,
                     u0 = [(x, y) -> exp(-30 * ((x - 0.5)^2 + (y - 0.65)^2))],
                     stabilization = LDGStabilization(10.0, 0.0))
sol_rot  = solve(prob_rot, RK4(); dt = 1e-3, tfinal = 1.0)
extrema(sol_rot.u)

# The bump has rotated about the center while diffusing — with a Makie
# backend loaded, `scaplot(mesh, sol_rot.u[:, 1, :], show_mesh = true)`
# shows it. For run-time monitoring of exactly this kind of solve
# (conservation drift, in-loop L² errors, snapshots), see the
# [Callbacks and diagnostics](callbacks.md) tutorial.
