# The single implementation of face/element geometry at quadrature points
# (roadmap A2.1). `DGContext` (and through it the KA residual kernels and the
# batched HDG assembly) consume these; the hand-copied Jacobian/normal blocks
# still living in the legacy `rinvexpl`/`rldgexpl`/`getq`/HDG local solvers
# disappear with them when the legacy path is retired (A2.2).
module Geometry

using LinearAlgebra

export RefTables, face_geometry!, element_geometry!

"""
    RefTables(master)

Reference-element tabulations shared by all geometry evaluations: shape
values/derivatives at volume and face quadrature points, quadrature weights,
weighted derivative tables and the reference mass matrix.
"""
struct RefTables{M <: AbstractMatrix{Float64}, V <: AbstractVector{Float64}}
    shap   :: M   # (npl, ng)   values at volume quadrature points
    shapξ  :: M   # (npl, ng)   d/dξ
    shapη  :: M   # (npl, ng)   d/dη
    shapξg :: M   # shapξ * Diagonal(gwgh)
    shapηg :: M   # shapη * Diagonal(gwgh)
    gwgh   :: V   # (ng,)       volume quadrature weights
    mass   :: M   # (npl, npl)  reference mass matrix
    sh1d   :: M   # (np1d, ng1d) values at face quadrature points
    sh1dξ  :: M   # (np1d, ng1d) reference derivatives
    gw1d   :: V   # (ng1d,)     face quadrature weights
end

function RefTables(master)
    shap = master.shap[:, 1, :]
    shapξ = master.shap[:, 2, :]
    shapη = master.shap[:, 3, :]
    return RefTables(shap, shapξ, shapη,
                     shapξ * Diagonal(master.gwgh), shapη * Diagonal(master.gwgh),
                     collect(master.gwgh), Matrix(master.mass),
                     master.sh1d[:, 1, :], master.sh1d[:, 2, :],
                     collect(master.gw1d))
end

"""
    face_geometry!(nlg, dws, pfg, rt, coords; edge=nothing)

Fill, for one face with high-order node coordinates `coords (np1d, 2)`
(ordered left-element-outward, i.e. `dgnodes[perml, :, el]`):

- `nlg (ng1d, 2)`: outward unit normal w.r.t. the left element,
- `dws (ng1d,)`: quadrature-weighted face measure `gw1d .* dsdξ`,
- `pfg (ng1d, 2)`: physical coordinates of the face quadrature points.

For a straight face pass `edge = (dx1, dx2)`, the vertex-to-vertex edge
vector; the metric is then constant. Otherwise the metric is evaluated from
the high-order nodes (curved face).
"""
function face_geometry!(nlg, dws, pfg, rt::RefTables, coords; edge=nothing)
    ng1d = length(rt.gw1d)
    if edge === nothing
        for g in 1:ng1d
            τ1 = dot(@view(rt.sh1dξ[:, g]), @view(coords[:, 1]))
            τ2 = dot(@view(rt.sh1dξ[:, g]), @view(coords[:, 2]))
            τn = sqrt(τ1^2 + τ2^2)
            nlg[g, 1] = τ2 / τn
            nlg[g, 2] = -τ1 / τn
            dws[g] = rt.gw1d[g] * τn
        end
    else
        dx1, dx2 = edge
        dsdξ = sqrt(dx1^2 + dx2^2)
        for g in 1:ng1d
            nlg[g, 1] = dx2 / dsdξ
            nlg[g, 2] = -dx1 / dsdξ
            dws[g] = rt.gw1d[g] * dsdξ
        end
    end
    pfg .= rt.sh1d' * coords
    return nothing
end

"""
    element_geometry!(shapx, shapy, wjac, pg, rt, coords; verts=nothing) -> M

Fill, for one element with high-order node coordinates `coords (npl, 2)`:

- `shapx, shapy (npl, ng)`: quadrature- and Jacobian-weighted physical
  derivative tables (`∫ ∂φ/∂x ⋅ f` becomes `shapx * f(quad)`),
- `wjac (ng,)`: `gwgh .* detJ`,
- `pg (ng, 2)`: physical coordinates of the volume quadrature points,

and return the element mass matrix `M (npl, npl)`.

For a straight (affine) element pass `verts`, the `(3, 2)` vertex coordinate
matrix; the Jacobian is then constant and `M` is the scaled reference mass.
Otherwise the isoparametric map is evaluated per quadrature point (curved
element).
"""
function element_geometry!(shapx, shapy, wjac, pg, rt::RefTables, coords; verts=nothing)
    if verts !== nothing
        xξ = verts[2, 1] - verts[1, 1]
        xη = verts[3, 1] - verts[1, 1]
        yξ = verts[2, 2] - verts[1, 2]
        yη = verts[3, 2] - verts[1, 2]
        detJ = xξ * yη - xη * yξ
        shapx .= rt.shapξg .* yη .- rt.shapηg .* yξ
        shapy .= .-rt.shapξg .* xη .+ rt.shapηg .* xξ
        wjac .= rt.gwgh .* detJ
        M = rt.mass .* detJ
    else
        ng = length(rt.gwgh)
        for j in 1:ng
            J = hcat(@view(rt.shapξ[:, j]), @view(rt.shapη[:, j]))' * coords
            invJ = inv(J)
            dJ = det(J)
            shap∇ = invJ * hcat(@view(rt.shapξ[:, j]), @view(rt.shapη[:, j]))'
            shapx[:, j] .= shap∇[1, :] .* rt.gwgh[j] .* dJ
            shapy[:, j] .= shap∇[2, :] .* rt.gwgh[j] .* dJ
            wjac[j] = rt.gwgh[j] * dJ
        end
        M = rt.shap * Diagonal(wjac) * rt.shap'
    end
    pg .= rt.shap' * coords
    return M
end

end # module Geometry
