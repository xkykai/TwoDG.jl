# The single implementation of face/element geometry at quadrature points:
# the pointwise kernels (`face_geometry!`/`element_geometry!`) and the
# precomputed, device-ready caches built from them (`GeometricFactors`,
# `SideGeometry`). Every method (CG/DG/HDG) consumes these — no solver may
# re-derive Jacobians, normals, or mass matrices on its own.
module Geometry

using LinearAlgebra
using Adapt
import KernelAbstractions

export RefTables, face_geometry!, element_geometry!,
       GeometricFactors, SideGeometry

"""
    RefTables(master)

Reference-element tabulations shared by all geometry evaluations: shape
values/derivatives at volume and face quadrature points, quadrature weights,
weighted derivative tables and the reference mass matrix. The element type
follows the master element's.
"""
struct RefTables{T, M <: AbstractMatrix{T}, V <: AbstractVector{T}}
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

"""
    GeometricFactors(master, mesh; T=Float64)

One-time precomputation of all mesh geometry at quadrature points, shared by
every discretization (the DG residual kernels consume it directly as
`DGContext`; the HDG/CG caches compose it): face/element connectivity
resolved to plain index arrays (no runtime `findfirst`), face and element
geometry evaluated at quadrature points (straight and curved elements handled
uniformly), and explicit inverse mass matrices. All fields are plain arrays of
eltype `T` (`Int32` for indices), so the whole cache moves to a GPU with
`Adapt.adapt(CuArray, gf)`.

Faces `1:ni` are interior, `ni+1:nf` are boundary (the ordering of `mesh.f`).

Field shapes (npl volume nodes, np1d face nodes, ng/ng1d quadrature points):

- `facecon (np1d, 2, nf)`: volume-node indices of face nodes; side 1 = left
  element, side 2 = right (unused for boundary faces).
- `f_el (nf, 2)`: (left element, right element); column 2 is `-ib` (negative
  boundary tag) for boundary faces, as in `mesh.f`.
- `nlg (ng1d, 2, nf)`, `dws (ng1d, nf)`, `pfg (ng1d, 2, nf)`: outward unit
  normal (w.r.t. left element), weighted measure `gw1d .* dsdxi`, and physical
  coordinates at face quadrature points.
- `shapx, shapy (npl, ng, nt)`: quadrature- and Jacobian-weighted physical
  derivative matrices; `wjac (ng, nt)`: `gwgh .* detJ`; `pg (ng, 2, nt)`:
  physical coordinates of volume quadrature points.
- `Minv (npl, npl, nt)`: inverse element mass matrices.
- `sh1d (np1d, ng1d)`, `shap (npl, ng)`: shape-function values (shared).
"""
struct GeometricFactors{T, A2 <: AbstractMatrix{T}, A3 <: AbstractArray{T, 3},
                        I2 <: AbstractMatrix{Int32}, I3 <: AbstractArray{Int32, 3}}
    ni      :: Int
    nf      :: Int
    nt      :: Int
    npl     :: Int
    np1d    :: Int
    ng      :: Int
    ng1d    :: Int
    facecon :: I3
    f_el    :: I2
    nlg     :: A3
    dws     :: A2
    pfg     :: A3
    shapx   :: A3
    shapy   :: A3
    wjac    :: A2
    pg      :: A3
    Minv    :: A3
    sh1d    :: A2
    shap    :: A2
end

Adapt.@adapt_structure GeometricFactors

KernelAbstractions.get_backend(gf::GeometricFactors) = KernelAbstractions.get_backend(gf.dws)

Base.eltype(::GeometricFactors{T}) where {T} = T

function GeometricFactors(master, mesh; T::Type{<:AbstractFloat}=Float64)
    p, t, f, t2f = mesh.p, mesh.t, mesh.f, mesh.t2f
    nt = size(t, 1)
    nf = size(f, 1)
    npl = size(mesh.dgnodes, 1)
    perm = master.perm
    np1d = size(perm, 1)
    ng1d, ng = length(master.gw1d), length(master.gwgh)

    rt = RefTables(master)

    ni = something(findfirst(i -> f[i, 4] < 0, 1:nf), nf + 1) - 1

    # --- faces: connectivity + geometry at 1D quadrature points ---
    facecon = ones(Int32, np1d, 2, nf)
    f_el = ones(Int32, nf, 2)
    nlg = zeros(ng1d, 2, nf)
    dws = zeros(ng1d, nf)
    pfg = zeros(ng1d, 2, nf)

    for i in 1:nf
        ipt = f[i, 1] + f[i, 2]
        el = f[i, 3]
        ipl = sum(@view t[el, :]) - ipt
        isl = findfirst(==(ipl), @view t[el, :])
        iol = t2f[el, isl] < 0 ? 2 : 1
        perml = perm[:, isl, iol]
        facecon[:, 1, i] .= perml
        f_el[i, 1] = el
        f_el[i, 2] = f[i, 4]

        if f[i, 4] > 0
            er = f[i, 4]
            ipr = sum(@view t[er, :]) - ipt
            isr = findfirst(==(ipr), @view t[er, :])
            ior = t2f[er, isr] < 0 ? 2 : 1
            facecon[:, 2, i] .= perm[:, isr, ior]
        end

        coords = mesh.dgnodes[perml, :, el]   # (np1d, 2)
        edge = mesh.fcurved[i] ? nothing :
               (p[f[i, 2], 1] - p[f[i, 1], 1], p[f[i, 2], 2] - p[f[i, 1], 2])
        face_geometry!(@view(nlg[:, :, i]), @view(dws[:, i]), @view(pfg[:, :, i]),
                       rt, coords; edge)
    end

    # --- elements: geometry at 2D quadrature points + inverse mass ---
    shapx = zeros(npl, ng, nt)
    shapy = zeros(npl, ng, nt)
    wjac = zeros(ng, nt)
    pg = zeros(ng, 2, nt)
    Minv = zeros(npl, npl, nt)

    for i in 1:nt
        verts = mesh.tcurved[i] ? nothing : p[t[i, :], :]
        M = element_geometry!(@view(shapx[:, :, i]), @view(shapy[:, :, i]),
                              @view(wjac[:, i]), @view(pg[:, :, i]),
                              rt, mesh.dgnodes[:, :, i]; verts)
        Minv[:, :, i] .= inv(M)
    end

    return GeometricFactors(ni, nf, nt, npl, np1d, ng, ng1d,
                            facecon, f_el,
                            T.(nlg), T.(dws), T.(pfg),
                            T.(shapx), T.(shapy), T.(wjac), T.(pg), T.(Minv),
                            T.(rt.sh1d), T.(rt.shap))
end

"""
    SideGeometry(master, mesh; T=Float64)

Face geometry in *element-local* orientation, one entry per (local side,
element) — what the HDG trace assembly needs (its trace basis follows the
element's own counterclockwise traversal `perm[:, s, 1]`, not the global
face's stored direction):

- `nl (ng1d, 2, 3, nt)`: outward unit normal of side `s` of element `e`,
- `sw (ng1d, 3, nt)`: quadrature-weighted side measure `gw1d .* dsdξ`,
- `pfs (ng1d, 2, 3, nt)`: physical coordinates of the side quadrature points.

Evaluated isoparametrically from the side's high-order nodes, so curved
faces are exact to the geometric order.
"""
struct SideGeometry{T, A3 <: AbstractArray{T, 3}, A4 <: AbstractArray{T, 4}}
    nl  :: A4
    sw  :: A3
    pfs :: A4
end

Adapt.@adapt_structure SideGeometry

Base.eltype(::SideGeometry{T}) where {T} = T

function SideGeometry(master, mesh; T::Type{<:AbstractFloat}=Float64)
    rt = RefTables(master)
    perm = master.perm
    ng1d = length(rt.gw1d)
    nt = size(mesh.t, 1)

    nl = zeros(ng1d, 2, 3, nt)
    sw = zeros(ng1d, 3, nt)
    pfs = zeros(ng1d, 2, 3, nt)

    for e in 1:nt, s in 1:3
        coords = mesh.dgnodes[perm[:, s, 1], :, e]
        face_geometry!(@view(nl[:, :, s, e]), @view(sw[:, s, e]),
                       @view(pfs[:, :, s, e]), rt, coords)
    end

    return SideGeometry(T.(nl), T.(sw), T.(pfs))
end

end # module Geometry
