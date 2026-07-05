# The single implementation of face/element geometry at quadrature points:
# the pointwise kernels (`face_geometry!`/`element_geometry!`) and the
# precomputed, device-ready caches built from them (`GeometricFactors`,
# `SideGeometry`). Every method (CG/DG/HDG) consumes these — no solver may
# re-derive Jacobians, normals, or mass matrices on its own.
#
# All of it is dimension-generic (THREED_PLAN Phase A): direction is an array
# axis (`shapd (npl, ng, Dim, …)`), the normal comes from `face_normal`
# dispatch (tangent rotation in 2D, cross product in 3D), and Jacobians are
# `Dim × Dim`. There are no `_2d`/`_3d` forks.
module Geometry

using LinearAlgebra
using StaticArrays
using Adapt
import KernelAbstractions

export RefTables, face_geometry!, element_geometry!, face_normal,
       GeometricFactors, SideGeometry

"""
    face_normal(τ) -> SVector{2}
    face_normal(τ₁, τ₂) -> SVector{3}

Outward (unnormalized) face normal from the face tangent vector(s): the
quarter-turn rotation of the single tangent in 2D, the cross product of the
two tangents in 3D. Its norm is the face measure Jacobian. This is the one
place where 2D and 3D geometry genuinely differ.
"""
@inline face_normal(τ::SVector{2}) = SVector(τ[2], -τ[1])
@inline face_normal(τ₁::SVector{3}, τ₂::SVector{3}) = cross(τ₁, τ₂)

"""
    adjugate(J) -> SMatrix

Adjugate (transposed cofactor matrix) of a `Dim × Dim` Jacobian:
`adjugate(J) == det(J) * inv(J)` without the division — the exact-arithmetic
form the affine geometry path contracts reference derivatives with.
"""
@inline adjugate(J::SMatrix{2, 2}) = @SMatrix [J[2, 2] -J[1, 2]; -J[2, 1] J[1, 1]]
@inline adjugate(J::SMatrix{3, 3}) =
    @SMatrix [J[2, 2]*J[3, 3]-J[2, 3]*J[3, 2]  J[1, 3]*J[3, 2]-J[1, 2]*J[3, 3]  J[1, 2]*J[2, 3]-J[1, 3]*J[2, 2];
              J[2, 3]*J[3, 1]-J[2, 1]*J[3, 3]  J[1, 1]*J[3, 3]-J[1, 3]*J[3, 1]  J[1, 3]*J[2, 1]-J[1, 1]*J[2, 3];
              J[2, 1]*J[3, 2]-J[2, 2]*J[3, 1]  J[1, 2]*J[3, 1]-J[1, 1]*J[3, 2]  J[1, 1]*J[2, 2]-J[1, 2]*J[2, 1]]

"""
    RefTables(master)

Reference-element tabulations shared by all geometry evaluations: shape
values/derivatives at volume and face quadrature points, quadrature weights,
weighted derivative tables and the reference mass matrix. Dimension-generic:
the face tables come from the master element's recursive `face` element.
"""
struct RefTables{Dim, T, M <: AbstractMatrix{T}, A3 <: AbstractArray{T, 3},
                 V <: AbstractVector{T}}
    shap   :: M    # (npl, ng)        values at volume quadrature points
    shapd  :: A3   # (npl, ng, Dim)   reference derivatives d/dξ_d
    shapdg :: A3   # shapd .* gwgh'   (quadrature-weighted derivatives)
    gwgh   :: V    # (ng,)            volume quadrature weights
    mass   :: M    # (npl, npl)       reference mass matrix
    fshap  :: M    # (npf, ngf)       face values at face quadrature points
    fshapd :: A3   # (npf, ngf, Dim-1) face reference derivatives
    gwf    :: V    # (ngf,)           face quadrature weights
end

Base.ndims(::RefTables{Dim}) where {Dim} = Dim

function RefTables(master)
    Dim = ndims(master)
    npl, ng = size(master.shap, 1), size(master.shap, 3)
    T = eltype(master.shap)

    shap = master.shap[:, 1, :]
    shapd = Array{T, 3}(undef, npl, ng, Dim)
    shapdg = Array{T, 3}(undef, npl, ng, Dim)
    for d in 1:Dim
        shapd[:, :, d] .= master.shap[:, 1 + d, :]
        shapdg[:, :, d] .= @view(shapd[:, :, d]) * Diagonal(master.gwgh)
    end

    fe = master.face
    npf, ngf = size(fe.shap, 1), size(fe.shap, 3)
    fshap = fe.shap[:, 1, :]
    fshapd = Array{T, 3}(undef, npf, ngf, Dim - 1)
    for d in 1:(Dim - 1)
        fshapd[:, :, d] .= fe.shap[:, 1 + d, :]
    end

    return RefTables{Dim, T, Matrix{T}, Array{T, 3}, Vector{T}}(
        shap, shapd, shapdg, collect(master.gwgh), Matrix(master.mass),
        fshap, fshapd, collect(fe.gwgh))
end

"""
    face_geometry!(nlg, dws, pfg, rt, coords; tangent=nothing)

Fill, for one face with high-order node coordinates `coords (npf, Dim)`
(ordered left-element-outward, i.e. `dgnodes[perml, :, el]`):

- `nlg (ngf, Dim)`: outward unit normal w.r.t. the left element,
- `dws (ngf,)`: quadrature-weighted face measure,
- `pfg (ngf, Dim)`: physical coordinates of the face quadrature points.

For a straight face pass `tangent`, an `NTuple{Dim-1, SVector{Dim}}` of
vertex-to-vertex edge vectors; the metric is then constant. Otherwise the
metric is evaluated from the high-order nodes (curved face).
"""
function face_geometry!(nlg, dws, pfg, rt::RefTables{Dim}, coords;
                        tangent=nothing) where {Dim}
    ngf = length(rt.gwf)
    if tangent === nothing
        for g in 1:ngf
            τ = ntuple(Val(Dim - 1)) do k
                SVector{Dim}(ntuple(Val(Dim)) do d
                    dot(@view(rt.fshapd[:, g, k]), @view(coords[:, d]))
                end)
            end
            nvec = face_normal(τ...)
            τn = norm(nvec)
            for d in 1:Dim
                nlg[g, d] = nvec[d] / τn
            end
            dws[g] = rt.gwf[g] * τn
        end
    else
        nvec = face_normal(tangent...)
        τn = norm(nvec)
        for g in 1:ngf
            for d in 1:Dim
                nlg[g, d] = nvec[d] / τn
            end
            dws[g] = rt.gwf[g] * τn
        end
    end
    pfg .= rt.fshap' * coords
    return nothing
end

"""
    element_geometry!(shapd, wjac, pg, rt, coords; verts=nothing) -> M

Fill, for one element with high-order node coordinates `coords (npl, Dim)`:

- `shapd (npl, ng, Dim)`: quadrature- and Jacobian-weighted physical
  derivative tables (`∫ ∂φ/∂x_d ⋅ f` becomes `shapd[:, :, d] * f(quad)`),
- `wjac (ng,)`: `gwgh .* detJ`,
- `pg (ng, Dim)`: physical coordinates of the volume quadrature points,

and return the element mass matrix `M (npl, npl)`.

For a straight (affine) element pass `verts`, the `(Dim+1, Dim)` vertex
coordinate matrix; the Jacobian is then constant and `M` is the scaled
reference mass. Otherwise the isoparametric map is evaluated per quadrature
point (curved element).
"""
function element_geometry!(shapd, wjac, pg, rt::RefTables{Dim}, coords;
                           verts=nothing) where {Dim}
    if verts !== nothing
        # affine map: constant Jacobian J[d, k] = ∂x_d/∂ξ_k from the vertices;
        # physical derivatives contract the weighted reference tables with the
        # adjugate (so detJ divides out of ∫ ∂φ/∂x ⋅ detJ)
        J = SMatrix{Dim, Dim}(ntuple(Val(Dim * Dim)) do i
            d = (i - 1) % Dim + 1
            k = (i - 1) ÷ Dim + 1
            verts[1 + k, d] - verts[1, d]
        end)
        detJ = det(J)
        C = adjugate(J)
        for d in 1:Dim
            sd = @view shapd[:, :, d]
            sd .= @view(rt.shapdg[:, :, 1]) .* C[1, d]
            for k in 2:Dim
                sd .+= @view(rt.shapdg[:, :, k]) .* C[k, d]
            end
        end
        wjac .= rt.gwgh .* detJ
        M = rt.mass .* detJ
    else
        ng = length(rt.gwgh)
        for j in 1:ng
            Jref = @view(rt.shapd[:, j, :])' * coords   # (Dim, Dim): J[k, d] = ∂x_d/∂ξ_k
            invJ = inv(Jref)
            dJ = det(Jref)
            shap∇ = invJ * @view(rt.shapd[:, j, :])'    # (Dim, npl) physical derivatives
            for d in 1:Dim
                shapd[:, j, d] .= shap∇[d, :] .* rt.gwgh[j] .* dJ
            end
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

Field shapes (npl volume nodes, npf face nodes, ng/ngf quadrature points):

- `facecon (npf, 2, nf)`: volume-node indices of face nodes; side 1 = left
  element, side 2 = right (unused for boundary faces).
- `f_el (nf, 2)`: (left element, right element); column 2 is `-ib` (negative
  boundary tag) for boundary faces, as in `mesh.f`.
- `nlg (ngf, Dim, nf)`, `dws (ngf, nf)`, `pfg (ngf, Dim, nf)`: outward unit
  normal (w.r.t. left element), weighted measure, and physical coordinates at
  face quadrature points.
- `shapd (npl, ng, Dim, nt)`: quadrature- and Jacobian-weighted physical
  derivative tables; `wjac (ng, nt)`: `gwgh .* detJ`; `pg (ng, Dim, nt)`:
  physical coordinates of volume quadrature points.
- `Minv (npl, npl, nt)`: inverse element mass matrices.
- `shapf (npf, ngf)`, `shap (npl, ng)`: shape-function values (shared).

Deprecated property aliases (one release, 2D): `shapx`/`shapy` view
`shapd[:, :, 1/2, :]`; `sh1d`, `np1d`, `ng1d` read `shapf`, `npf`, `ngf`.
"""
struct GeometricFactors{T, Dim, A2 <: AbstractMatrix{T}, A3 <: AbstractArray{T, 3},
                        A4 <: AbstractArray{T, 4},
                        I2 <: AbstractMatrix{Int32}, I3 <: AbstractArray{Int32, 3}}
    ni      :: Int
    nf      :: Int
    nt      :: Int
    npl     :: Int
    npf     :: Int
    ng      :: Int
    ngf     :: Int
    facecon :: I3
    f_el    :: I2
    nlg     :: A3
    dws     :: A2
    pfg     :: A3
    shapd   :: A4
    wjac    :: A2
    pg      :: A3
    Minv    :: A3
    shapf   :: A2
    shap    :: A2
end

# positional constructor deriving the Dim type parameter from the data (the
# normal table's direction axis); this is also the form `@adapt_structure`
# calls when moving the cache to a device
function GeometricFactors(ni::Integer, nf::Integer, nt::Integer, npl::Integer,
                          npf::Integer, ng::Integer, ngf::Integer,
                          facecon::I3, f_el::I2, nlg::A3, dws::A2, pfg::A3,
                          shapd::A4, wjac::A2, pg::A3, Minv::A3,
                          shapf::A2, shap::A2) where
                         {A2 <: AbstractMatrix, A3 <: AbstractArray{<:Any, 3},
                          A4 <: AbstractArray{<:Any, 4},
                          I2 <: AbstractMatrix{Int32}, I3 <: AbstractArray{Int32, 3}}
    Dim = size(nlg, 2)
    return GeometricFactors{eltype(dws), Dim, A2, A3, A4, I2, I3}(
        ni, nf, nt, npl, npf, ng, ngf, facecon, f_el, nlg, dws, pfg,
        shapd, wjac, pg, Minv, shapf, shap)
end

Adapt.@adapt_structure GeometricFactors

KernelAbstractions.get_backend(gf::GeometricFactors) = KernelAbstractions.get_backend(gf.dws)

Base.eltype(::GeometricFactors{T}) where {T} = T
Base.ndims(::GeometricFactors{T, Dim}) where {T, Dim} = Dim

# pre-Dim field names, kept as property aliases for one release (NEWS.md)
@inline function Base.getproperty(gf::GeometricFactors, s::Symbol)
    s === :shapx && return view(getfield(gf, :shapd), :, :, 1, :)
    s === :shapy && return view(getfield(gf, :shapd), :, :, 2, :)
    s === :sh1d  && return getfield(gf, :shapf)
    s === :np1d  && return getfield(gf, :npf)
    s === :ng1d  && return getfield(gf, :ngf)
    return getfield(gf, s)
end

# vertex-to-vertex tangent vectors of a straight face, from the face's stored
# vertex list (left-element-outward order)
@inline straight_face_tangents(p, fv, ::Val{2}) =
    (SVector(p[fv[2], 1] - p[fv[1], 1], p[fv[2], 2] - p[fv[1], 2]),)
@inline straight_face_tangents(p, fv, ::Val{3}) =
    (SVector(p[fv[2], 1] - p[fv[1], 1], p[fv[2], 2] - p[fv[1], 2], p[fv[2], 3] - p[fv[1], 3]),
     SVector(p[fv[3], 1] - p[fv[1], 1], p[fv[3], 2] - p[fv[1], 2], p[fv[3], 3] - p[fv[1], 3]))

function GeometricFactors(master, mesh; T::Type{<:AbstractFloat}=Float64)
    Dim = ndims(master)
    p, t, f, t2f, t2o = mesh.p, mesh.t, mesh.f, mesh.t2f, mesh.t2o
    nt = size(t, 1)
    nf = size(f, 1)
    npl = size(mesh.dgnodes, 1)
    perm = master.perm
    npf = size(perm, 1)
    nv = Dim + 1                        # vertices per element = faces per element
    rt = RefTables(master)
    ng, ngf = length(rt.gwgh), length(rt.gwf)

    ni = something(findfirst(i -> f[i, nv + 1] < 0, 1:nf), nf + 1) - 1

    # --- faces: connectivity + geometry at face quadrature points ---
    facecon = ones(Int32, npf, 2, nf)
    f_el = ones(Int32, nf, 2)
    nlg = zeros(ngf, Dim, nf)
    dws = zeros(ngf, nf)
    pfg = zeros(ngf, Dim, nf)

    for i in 1:nf
        ipt = sum(@view f[i, 1:Dim])
        el = f[i, nv]
        ipl = sum(@view t[el, :]) - ipt
        isl = findfirst(==(ipl), @view t[el, :])
        iol = t2o[el, isl]
        perml = perm[:, isl, iol]
        facecon[:, 1, i] .= perml
        f_el[i, 1] = el
        f_el[i, 2] = f[i, nv + 1]

        if f[i, nv + 1] > 0
            er = f[i, nv + 1]
            ipr = sum(@view t[er, :]) - ipt
            isr = findfirst(==(ipr), @view t[er, :])
            ior = t2o[er, isr]
            facecon[:, 2, i] .= perm[:, isr, ior]
        end

        coords = mesh.dgnodes[perml, :, el]   # (npf, Dim)
        tangent = mesh.fcurved[i] ? nothing :
                  straight_face_tangents(p, view(f, i, 1:Dim), Val(Dim))
        face_geometry!(@view(nlg[:, :, i]), @view(dws[:, i]), @view(pfg[:, :, i]),
                       rt, coords; tangent)
    end

    # --- elements: geometry at volume quadrature points + inverse mass ---
    shapd = zeros(npl, ng, Dim, nt)
    wjac = zeros(ng, nt)
    pg = zeros(ng, Dim, nt)
    Minv = zeros(npl, npl, nt)

    for i in 1:nt
        verts = mesh.tcurved[i] ? nothing : p[t[i, :], :]
        M = element_geometry!(@view(shapd[:, :, :, i]),
                              @view(wjac[:, i]), @view(pg[:, :, i]),
                              rt, mesh.dgnodes[:, :, i]; verts)
        Minv[:, :, i] .= inv(M)
    end

    return GeometricFactors(ni, nf, nt, npl, npf, ng, ngf,
                            facecon, f_el,
                            T.(nlg), T.(dws), T.(pfg),
                            T.(shapd), T.(wjac), T.(pg), T.(Minv),
                            T.(rt.fshap), T.(rt.shap))
end

"""
    SideGeometry(master, mesh; T=Float64)

Face geometry in *element-local* orientation, one entry per (local side,
element) — what the HDG trace assembly needs (its trace basis follows the
element's own canonical traversal `perm[:, s, 1]`, not the global face's
stored direction):

- `nl (ngf, Dim, Dim+1, nt)`: outward unit normal of side `s` of element `e`,
- `sw (ngf, Dim+1, nt)`: quadrature-weighted side measure,
- `pfs (ngf, Dim, Dim+1, nt)`: physical coordinates of the side quadrature
  points.

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
    Dim = ndims(master)
    rt = RefTables(master)
    perm = master.perm
    ngf = length(rt.gwf)
    nt = size(mesh.t, 1)
    ns = Dim + 1

    nl = zeros(ngf, Dim, ns, nt)
    sw = zeros(ngf, ns, nt)
    pfs = zeros(ngf, Dim, ns, nt)

    for e in 1:nt, s in 1:ns
        coords = mesh.dgnodes[perm[:, s, 1], :, e]
        face_geometry!(@view(nl[:, :, s, e]), @view(sw[:, s, e]),
                       @view(pfs[:, :, s, e]), rt, coords)
    end

    return SideGeometry(T.(nl), T.(sw), T.(pfs))
end

end # module Geometry
