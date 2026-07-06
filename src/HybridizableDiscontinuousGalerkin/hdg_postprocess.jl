using TwoDG.Geometry: RefTables, element_geometry!
using TwoDG.Masters: shape2d, shape3d

"""
    match_geometry!(master, mesh, master1, mesh1) -> mesh1

Replace the node coordinates of `mesh1`'s curved elements with the degree-`p`
isoparametric map of `mesh` evaluated at `mesh1`'s reference nodes
(`master1.plocal`) — exact, since ``P_p ⊂ P_{p+1}``. After the call both
meshes describe the *same* discrete geometry.

This is a prerequisite for the `p+2` superconvergence of
[`hdg_postprocess`](@ref) (and [`hdg_ns_postprocess`](@ref)) on curved
elements: when `mesh` and `mesh1` project their boundary nodes independently,
their isoparametric maps differ by `O(h^{p+1})`, the p- and (p+1)-fields are
composed with different maps of the reference element, and `u*` degrades to
`O(h^{p+1})` (no gain over `u`). Straight elements are unaffected (both maps
are already the same affine map).
"""
function match_geometry!(master, mesh, master1, mesh1)
    any(mesh.tcurved) || return mesh1
    Dim = ndims(master)
    pts = master1.plocal[:, 2:(Dim + 1)]
    nfs = _shape_at(Val(Dim), master.porder, master.plocal, pts)
    Phi = nfs[:, 1, :]                  # (npl, npl1) p-basis values at p+1 nodes
    for e in axes(mesh.dgnodes, 3)
        mesh.tcurved[e] || continue
        mesh1.dgnodes[:, :, e] .= Phi' * mesh.dgnodes[:, :, e]
    end
    return mesh1
end

_shape_at(::Val{2}, porder, plocal, pts) = shape2d(porder, plocal, pts)
_shape_at(::Val{3}, porder, plocal, pts) = shape3d(porder, plocal, pts)

"""
    hdg_postprocess(master, mesh, master1, mesh1, uh, qh)

Postprocesses the HDG solution to obtain a superconvergent solution
(Nguyen, Peraire & Cockburn, JCP 230, 2011). Dimension-generic: works on
triangles and tetrahedra alike. On meshes with curved elements, `mesh1` must
carry the same discrete geometry as `mesh` — see [`match_geometry!`](@ref).

# Arguments
- `master`, `mesh`: master structure and mesh of order `porder`
- `master1`, `mesh1`: master structure and mesh of order `porder + 1`
  (built with the **same** quadrature rule as `master`)
- `uh (npl, 1, nt)`: approximate scalar variable
- `qh (npl, Dim, nt)`: approximate flux `q = -∇u` (pass `q ./ κ` when the
  solver returns `κ∇u`-scaled fluxes)

# Returns
- `ustarh (npl1, 1, nt)`: postprocessed scalar variable

# HDG Postprocessing
Solves, element by element, the local Neumann problem: find `u*` in
`P_{p+1}(K)` such that

- `(∇u*, ∇v)_K = -(qh, ∇v)_K` for all `v` in `P_{p+1}(K)`,
- the mean of `u*` over `K` equals the mean of `uh`,

which converges at order `p+2` for diffusion problems.
"""
function hdg_postprocess(master, mesh, master1, mesh1, uh, qh)
    Dim = ndims(master)
    npl1 = size(mesh1.dgnodes, 1)
    nt = size(mesh.dgnodes, 3)
    ng = length(master1.gwgh)
    # both masters must share one quadrature rule: the p-basis data (uh, qh)
    # is evaluated at the p+1 element's quadrature points below
    @assert length(master.gwgh) == ng

    rt1 = RefTables(master1)
    shap = master.shap[:, 1, :]        # p-basis values at the shared quad points

    ustarh = zeros(npl1, 1, nt)

    Threads.@threads for i in 1:nt
        # element geometry of the p+1 mesh: quadrature/Jacobian-weighted
        # physical derivative tables (straight elements via the affine path)
        shapd = Array{Float64, 3}(undef, npl1, ng, Dim)
        wjac = Vector{Float64}(undef, ng)
        pg = Matrix{Float64}(undef, ng, Dim)
        verts = mesh1.tcurved[i] ? nothing : mesh1.p[mesh1.t[i, :], :]
        element_geometry!(shapd, wjac, pg, rt1, @view(mesh1.dgnodes[:, :, i]); verts)

        # stiffness K = ∫ ∇φ·∇φ and moment rhs r = -∫ ∇v·qh; shapd carries one
        # factor of (gwgh ⋅ detJ), so one of the two tables is unweighted
        w = 1 ./ wjac
        K = zeros(npl1, npl1)
        r = zeros(npl1)
        for d in 1:Dim
            sd = @view shapd[:, :, d]
            K .+= sd * Diagonal(w) * sd'
            qg = shap' * @view(qh[:, d, i])
            r .-= sd * qg
        end

        # mean constraint replaces the last equation: ∫ u* = ∫ uh
        K[end, :] .= vec(sum(rt1.shap * Diagonal(wjac); dims=2))
        r[end] = sum((shap' * @view(uh[:, 1, i])) .* wjac)

        ustarh[:, 1, i] .= K \ r
    end

    return ustarh
end
