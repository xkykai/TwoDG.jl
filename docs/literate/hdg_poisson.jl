# # HDG Poisson and superconvergence
#
# This tutorial solves the Poisson problem with the hybridizable
# discontinuous Galerkin (HDG) method and demonstrates its signature
# property: a cheap, element-local postprocessing step that recovers a
# solution converging **one order faster** than the discretization itself
# (order ``p+2`` instead of ``p+1``).
#
# We use the manufactured solution
# ``u(x, y) = \sin(2\pi x)\,\sin(2\pi y)`` on the unit square with
# homogeneous Dirichlet boundaries, so ``-\Delta u = f`` with
# ``f = 8\pi^2 u``.

using TwoDG
using Printf

exact(x, y) = sin(2π * x) * sin(2π * y)

# HDG takes its source at quadrature points: a function of the coordinate
# matrix `p (nq, 2)` returning one value per point. Supplying the source this
# way (rather than interpolating it at solution nodes) is one of the two data
# rules that superconvergence depends on — the other is L2-projected (here:
# exactly zero) Dirichlet data.

source(p) = 8π^2 .* sin.(2π .* p[:, 1]) .* sin.(2π .* p[:, 2]);

# ## A single solve
#
# Build a mesh, wrap equation + boundary condition in an
# [`HDGProblem`](@ref), and `solve`. The default algorithm is a
# block-Jacobi-preconditioned GMRES on the statically condensed trace
# system; [`Direct`](@ref) assembles the same system and factorizes it
# sparsely.

mesh = mkmesh_square(17, 17, 2, 0, 1)
prob = HDGProblem(PoissonEquation(), mesh; bc = Dirichlet(0.0), source)
sol  = solve(prob, Direct())
l2error(sol, exact)

# The solution object also carries the flux approximation
# `sol.q (npl, 2, nt)` and the face traces `sol.uhat` — we need `q` for the
# postprocessing below.

# ## Postprocessing
#
# HDG approximates both ``u`` and its flux ``q = -\kappa \nabla u`` to order
# ``p+1``. [`hdg_postprocess`](@ref) solves, element by element, the tiny
# local problem: find ``u^* \in P_{p+1}(K)`` with
# ``\nabla u^* = -q_h`` and ``\bar{u}^* = \bar{u}_h``. Because `q` is already
# accurate to order ``p+1``, ``u^*`` gains an order — with no global solve.
#
# The postprocessed field lives in the ``p+1`` space, so it needs a ``p+1``
# mesh and [`ReferenceElement`](@ref) for its representation. The two
# reference elements must share one quadrature rule (`pgauss`, the
# polynomial degree integrated exactly), because the postprocessing
# evaluates the ``p``-basis data at the ``p+1`` element's quadrature points:

porder = 2
pgauss = 4 * (porder + 1)
master  = ReferenceElement(mesh, pgauss)
mesh1   = mkmesh_square(17, 17, porder + 1, 0, 1)
master1 = ReferenceElement(mesh1, pgauss)

ustar = hdg_postprocess(master, mesh, master1, mesh1, sol.u, sol.q)
l2error(mesh1, ustar[:, 1, :], exact) / l2error(sol, exact)

# The postprocessed error is a small fraction of the plain error on the same
# mesh — and the gap widens under refinement, because the *rates* differ.

# ## Convergence study
#
# Refine the mesh at fixed order and measure both errors. `u` should
# converge at order ``p+1`` and ``u^*`` at ``p+2``.

function convergence(porder, ms)
    errs_u, errs_us = Float64[], Float64[]
    for m in ms
        mesh  = mkmesh_square(m, m, porder, 0, 1)
        mesh1 = mkmesh_square(m, m, porder + 1, 0, 1)
        pgauss = 4 * (porder + 1)
        master, master1 = ReferenceElement(mesh, pgauss), ReferenceElement(mesh1, pgauss)

        sol = solve(HDGProblem(PoissonEquation(), mesh;
                               bc = Dirichlet(0.0), source), Direct())
        ustar = hdg_postprocess(master, mesh, master1, mesh1, sol.u, sol.q)

        push!(errs_u, l2error(sol, exact))
        push!(errs_us, l2error(mesh1, ustar[:, 1, :], exact))
    end
    return errs_u, errs_us
end

ms = [5, 9, 17]
for porder in (1, 2)
    errs_u, errs_us = convergence(porder, ms)
    @printf "p = %d      h        ‖u - uₕ‖      rate    ‖u - u*‖      rate\n" porder
    for i in eachindex(ms)
        h = 1 / (ms[i] - 1)
        if i == 1
            @printf "      %8.4f   %11.3e      --   %11.3e      --\n" h errs_u[i] errs_us[i]
        else
            r  = log2(errs_u[i-1] / errs_u[i])
            rs = log2(errs_us[i-1] / errs_us[i])
            @printf "      %8.4f   %11.3e   %5.2f   %11.3e   %5.2f\n" h errs_u[i] r errs_us[i] rs
        end
    end
    println()
end

# The measured rates approach ``p+1`` for ``u_h`` and ``p+2`` for ``u^*``:
# the postprocessed ``p = 1`` solution is as accurate as a plain ``p = 2``
# solve, at a fraction of the globally coupled unknowns.
#
# ## Notes
#
# - The stabilization parameter (`stabilization` keyword of
#   [`HDGProblem`](@ref), default `1.0`) affects the constants and, for
#   ``q``, the rates; ``\tau = O(1)`` is the safe choice for
#   diffusion-dominated problems.
# - To visualize the solution, load a Makie backend and call
#   `scaplot(mesh1, ustar[:, 1, :], show_mesh = true)`.
