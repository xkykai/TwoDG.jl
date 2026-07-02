using LinearAlgebra
using Adapt
import KernelAbstractions
using ..Geometry: RefTables, face_geometry!, element_geometry!

"""
    DGContext{T}(master, mesh; T=Float64)

One-time precomputation of everything the KernelAbstractions residual path
needs: face/element connectivity resolved to plain index arrays (no runtime
`findfirst`), face and element geometry evaluated at quadrature points
(straight and curved elements handled uniformly), and explicit inverse mass
matrices. All fields are plain arrays of eltype `T` (`Int32` for indices), so
the whole context moves to a GPU with `Adapt.adapt(CuArray, ctx)`.

Faces `1:ni` are interior, `ni+1:nf` are boundary (the ordering of `mesh.f`).

Field shapes (npl volume nodes, np1d face nodes, ng/ng1d quadrature points):

- `facecon (np1d, 2, nf)`: volume-node indices of face nodes; side 1 = left
  element (`perml` of the legacy code), side 2 = right (`permr`, unused for
  boundary faces).
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
struct DGContext{T, A2 <: AbstractMatrix{T}, A3 <: AbstractArray{T, 3},
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

Adapt.@adapt_structure DGContext

KernelAbstractions.get_backend(ctx::DGContext) = KernelAbstractions.get_backend(ctx.dws)

Base.eltype(::DGContext{T}) where {T} = T

function DGContext(master, mesh; T::Type{<:AbstractFloat}=Float64)
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

    return DGContext(ni, nf, nt, npl, np1d, ng, ng1d,
                     facecon, f_el,
                     T.(nlg), T.(dws), T.(pfg),
                     T.(shapx), T.(shapy), T.(wjac), T.(pg), T.(Minv),
                     T.(rt.sh1d), T.(rt.shap))
end
