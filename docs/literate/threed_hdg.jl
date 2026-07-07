# # HDG superconvergence in 3D
#
# The hybridizable DG method carries its signature property to tetrahedra:
# a cheap element-local postprocessing recovers a solution one order more
# accurate than the discretization. This tutorial solves
#
# ```math
# -\Delta u = 3\pi^2 \sin(\pi x)\sin(\pi y)\sin(\pi z), \qquad
# u = 0 \text{ on } \partial\Omega,
# ```
#
# on the unit cube (exact solution
# ``u = \sin(\pi x)\sin(\pi y)\sin(\pi z)``) and measures both rates.

using TwoDG
using Printf

exact(x, y, z) = sin(π * x) * sin(π * y) * sin(π * z)
source(p) = reshape(3π^2 .* sin.(π .* p[:, 1]) .* sin.(π .* p[:, 2]) .*
                    sin.(π .* p[:, 3]), :, 1);

# The two superconvergence data rules from 2D apply verbatim: the source is
# supplied at quadrature points (a function of the coordinate matrix), and
# the Dirichlet data must be L2-projected — here it is exactly zero.
#
# ## Solve and postprocess
#
# On tetrahedra the trace unknowns live on triangular faces; the statically
# condensed system couples only those. The postprocessing needs a `p+1`
# mesh/master pair built with the *same* quadrature rule:

porder = 2
ngauss = 4 * (porder + 1)

errs_u, errs_us = Float64[], Float64[]
ns = (3, 5)
for n in ns
    mesh = mkmesh_box(n, n, n, porder)
    prob = HDGProblem(PoissonEquation{3}(), mesh; bc = Dirichlet(0.0), source)
    sol = solve(prob, Direct())

    master = ReferenceElement(mesh, ngauss)
    mesh1 = mkmesh_box(n, n, n, porder + 1)
    master1 = ReferenceElement(mesh1, ngauss)
    ustar = hdg_postprocess(master, mesh, master1, mesh1, sol.u, sol.q)

    push!(errs_u, l2error(sol, exact))
    push!(errs_us, l2error(mesh1, ustar[:, 1, :], exact))
    @printf "n = %d (%4d tets): |u - uh| = %.3e   |u - u*| = %.3e\n" n size(mesh.t, 1) errs_u[end] errs_us[end]
end
@printf "rates: u %.2f (design %d)   u* %.2f (design %d)\n" log2(errs_u[1] / errs_u[2]) porder + 1 log2(errs_us[1] / errs_us[2]) porder + 2

# ``u^*`` costs one tiny dense solve per element — no global system — and
# converges a full order faster. A postprocessed ``p = 2`` solution matches
# a plain ``p = 3`` solve that has ~1.75× more trace unknowns.
#
# ## Curved domains
#
# On curved (isoparametric) meshes two extra rules keep the rates:
#
# 1. **Shape regularity of the refinement family** — build curved meshes by
#    smoothly blending the reference geometry onto the boundary, not by
#    projecting boundary vertices after refinement (which stretches the
#    boundary layer like ``1/h`` and stalls every method's rate).
# 2. **The `p+1` mesh must carry the `p`-mesh's geometry** — call
#    [`match_geometry!`](@ref)`(master, mesh, master1, mesh1)` before
#    postprocessing. With independently projected maps the two geometries
#    differ by ``O(h^{p+1})`` and the superconvergence is destroyed.
#
# With both rules, the sphere-octant study in the test suite measures
# ``u`` at rate 2.85 and ``u^*`` at 3.60 for ``p = 2`` (14× more accurate
# at ``h = 1/8``). Full ``h^{p+2}`` superconvergence on curved domains
# additionally requires geometry one degree higher than the solution —
# with degree-``p`` geometry, ``u^*`` saturates at the geometry order
# ``O(h^{p+1})``, still one order better than nothing and strictly more
# accurate than ``u_h``.
#
# ## Navier-Stokes
#
# The same batched HDG engine solves steady and unsteady incompressible
# Navier-Stokes on tetrahedra (`hdg_ns_step_batched`, with Newton iteration
# and factorization-cache reuse); see
# `examples/hdg3d/runhdg3d_ns_beltrami.jl` for a genuinely 3D exact-solution
# benchmark (the Ethier-Steinman Beltrami flow, velocity rate ``p+1``).
