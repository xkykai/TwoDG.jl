# KernelAbstractions implementation of the LDG viscous path (gradient
# computation + viscous fluxes). Shares DGContext and the scatter/lift/mass
# kernels with rinvexpl_ka.jl. Dimension-generic like the inviscid path:
# gradients q are SMatrix{DIM, nc} (row d = ∂/∂x_d), staged face gradient
# fluxes live in one qfd (ngf, nc, Dim, nf) array, and the gradient/viscous
# loops run over the direction axis.
#
# LDG alternating traces: û on interior faces is the LEFT trace, and the
# viscous interface flux sees the RIGHT element's gradient. Gradient layout:
# q (npl, Dim, nc, nt).

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using Atomix
using StaticArrays
using Adapt

# acc + s .* v, elementwise on NTuples, without closure capture of a
# reassigned local (which would box it inside a kernel — a silent 100×
# performance cliff). Plain `+`/`*` so the FP rounding matches the unfused
# per-direction accumulation exactly.
@inline _axpy_tuple(acc::NTuple{D, T}, s::T, v::NTuple{D, T}) where {D, T} =
    ntuple(d -> acc[d] + s * v[d], Val(D))

# u interpolated to volume quadrature points (staged, reused by the gradient
# volume term and the viscous volume flux).
@kernel function _interp_u!(ug, @Const(u), @Const(shap), ::Val{NC}) where {NC}
    g, e = @index(Global, NTuple)
    T = eltype(ug)
    npl = size(shap, 1)

    @inbounds for c in 1:NC
        s = zero(T)
        for i in 1:npl
            s += shap[i, g] * u[i, c, e]
        end
        ug[g, c, e] = s
    end
end

# interpolate one side of a face to SVector / the gradient to SMatrix
@inline function _face_state(u, facecon, shapf, g, fc, side, e, ::Val{NC}) where {NC}
    T = eltype(u)
    npf = size(shapf, 1)
    return SVector{NC, T}(ntuple(Val(NC)) do c
        s = zero(T)
        @inbounds for j in 1:npf
            s += shapf[j, g] * u[facecon[j, side, fc], c, e]
        end
        s
    end)
end

@inline function _face_grad(q, facecon, shapf, g, fc, side, e,
                            ::Val{NC}, ::Val{DIM}) where {NC, DIM}
    T = eltype(q)
    npf = size(shapf, 1)
    return SMatrix{DIM, NC, T}(ntuple(Val(DIM * NC)) do k
        d = (k - 1) % DIM + 1
        c = (k - 1) ÷ DIM + 1
        s = zero(T)
        @inbounds for j in 1:npf
            s += shapf[j, g] * q[facecon[j, side, fc], d, c, e]
        end
        s
    end)
end

# LDG gradient, face term: û (left trace on interior faces, boundary_trace on
# the boundary) times weighted normal components -> qfd (ngf, nc, Dim, nf).
@kernel function _grad_face_flux!(qfd, @Const(u), @Const(facecon), @Const(f_el),
                                  @Const(nlg), @Const(dws), @Const(pfg), @Const(shapf),
                                  eq, bcs, time, ni, ::Val{NC}, ::Val{DIM}) where {NC, DIM}
    g, fc = @index(Global, NTuple)
    T = eltype(qfd)

    el = f_el[fc, 1]
    uL = _face_state(u, facecon, shapf, g, fc, 1, el, Val(NC))

    if fc <= ni
        û = uL
    else
        n = SVector{DIM, T}(ntuple(d -> @inbounds(nlg[g, d, fc]), Val(DIM)))
        x = SVector{DIM, T}(ntuple(d -> @inbounds(pfg[g, d, fc]), Val(DIM)))
        ib = -Int(f_el[fc, 2])
        û = apply_boundary_trace(bcs, ib, eq, uL, n, x, time)
    end

    w = dws[g, fc]
    @inbounds for d in 1:DIM, c in 1:NC
        qfd[g, c, d, fc] = û[c] * w * nlg[g, d, fc]
    end
end

# lift the face term with shapf and scatter into qtmp (+ left, - right)
@kernel function _grad_face_scatter!(qtmp, @Const(qfd), @Const(facecon),
                                     @Const(f_el), @Const(shapf), ni,
                                     ::Val{NC}, ::Val{DIM}) where {NC, DIM}
    j, fc = @index(Global, NTuple)
    T = eltype(qtmp)
    ngf = size(shapf, 2)

    el = f_el[fc, 1]
    il = facecon[j, 1, fc]
    interior = fc <= ni

    @inbounds for c in 1:NC
        # one fused g-loop: shapf[j, g] is read once for all directions
        acc = ntuple(_ -> zero(T), Val(DIM))
        for g in 1:ngf
            s = shapf[j, g]
            v = ntuple(d -> @inbounds(qfd[g, c, d, fc]), Val(DIM))
            acc = _axpy_tuple(acc, s, v)
        end
        for d in 1:DIM
            Atomix.@atomic qtmp[il, d, c, el] += acc[d]
        end
        if interior
            er = f_el[fc, 2]
            ir = facecon[j, 2, fc]
            for d in 1:DIM
                Atomix.@atomic qtmp[ir, d, c, er] -= acc[d]
            end
        end
    end
end

# LDG gradient, volume term: qtmp -= ∫ u ∇φ (element-local, no atomics)
@kernel function _grad_volume!(qtmp, @Const(ug), @Const(shapd),
                               ::Val{NC}, ::Val{DIM}) where {NC, DIM}
    i, e = @index(Global, NTuple)
    T = eltype(qtmp)
    ng = size(ug, 1)

    @inbounds for c in 1:NC
        # one fused g-loop: ug[g, c, e] is read once for all directions
        acc = ntuple(_ -> zero(T), Val(DIM))
        for g in 1:ng
            u = ug[g, c, e]
            v = ntuple(d -> @inbounds(shapd[i, g, d, e]), Val(DIM))
            acc = _axpy_tuple(acc, u, v)
        end
        for d in 1:DIM
            qtmp[i, d, c, e] -= acc[d]
        end
    end
end

@kernel function _grad_minv!(q, @Const(qtmp), @Const(Minv),
                             ::Val{NC}, ::Val{DIM}) where {NC, DIM}
    i, e = @index(Global, NTuple)
    T = eltype(q)
    npl = size(Minv, 1)

    @inbounds for c in 1:NC, d in 1:DIM
        acc = zero(T)
        for j in 1:npl
            acc += Minv[i, j, e] * qtmp[j, d, c, e]
        end
        q[i, d, c, e] = acc
    end
end

# face flux with viscous contribution: inviscid Riemann/boundary flux plus
# the LDG viscous numerical flux (interior, sees the RIGHT gradient per the
# LDG alternating choice — the left gradient is also interpolated and passed
# for generality) or the boundary viscous flux (sees the left gradient).
@kernel function _face_flux_visc!(fng, @Const(u), @Const(q), @Const(facecon),
                                  @Const(f_el), @Const(nlg), @Const(dws), @Const(pfg),
                                  @Const(shapf), eq, numflux, stab, bcs,
                                  time, ni, ::Val{NC}, ::Val{DIM}) where {NC, DIM}
    g, fc = @index(Global, NTuple)
    T = eltype(fng)

    el = f_el[fc, 1]
    n = SVector{DIM, T}(ntuple(d -> @inbounds(nlg[g, d, fc]), Val(DIM)))
    x = SVector{DIM, T}(ntuple(d -> @inbounds(pfg[g, d, fc]), Val(DIM)))

    uL = _face_state(u, facecon, shapf, g, fc, 1, el, Val(NC))

    if fc <= ni
        er = f_el[fc, 2]
        uR = _face_state(u, facecon, shapf, g, fc, 2, er, Val(NC))
        qL = _face_grad(q, facecon, shapf, g, fc, 1, el, Val(NC), Val(DIM))
        qR = _face_grad(q, facecon, shapf, g, fc, 2, er, Val(NC), Val(DIM))
        fn = numflux(eq, uL, uR, n, x, time) +
             viscous_numerical_flux(stab, eq, uL, uR, qL, qR, n, x, time)
    else
        ib = -Int(f_el[fc, 2])
        qL = _face_grad(q, facecon, shapf, g, fc, 1, el, Val(NC), Val(DIM))
        fn = apply_boundary_flux(bcs, ib, eq, numflux, uL, n, x, time) +
             apply_boundary_viscous_flux(bcs, ib, stab, eq, uL, qL, n, x, time)
    end

    w = dws[g, fc]
    @inbounds for c in 1:NC
        fng[g, c, fc] = fn[c] * w
    end
end

# volume flux with viscous contribution (reads the staged ug)
@kernel function _volume_flux_visc!(fdg, @Const(ug), @Const(q), @Const(shap),
                                    @Const(pg), eq, time,
                                    ::Val{NC}, ::Val{DIM}) where {NC, DIM}
    g, e = @index(Global, NTuple)
    T = eltype(fdg)
    npl = size(shap, 1)

    u = SVector{NC, T}(ntuple(c -> @inbounds(ug[g, c, e]), Val(NC)))
    qg = SMatrix{DIM, NC, T}(ntuple(Val(DIM * NC)) do k
        d = (k - 1) % DIM + 1
        c = (k - 1) ÷ DIM + 1
        s = zero(T)
        @inbounds for i in 1:npl
            s += shap[i, g] * q[i, d, c, e]
        end
        s
    end)
    x = SVector{DIM, T}(ntuple(d -> @inbounds(pg[g, d, e]), Val(DIM)))

    fdi = flux(eq, u, x, time)
    fdv = viscous_flux(eq, u, qg, x, time)
    @inbounds for d in 1:DIM, c in 1:NC
        fdg[g, c, d, e] = fdi[d][c] + fdv[d][c]
    end
end

"""
    RldgWorkspace(ctx, nc)

Staging buffers for the LDG residual path: everything in [`RinvWorkspace`](@ref)
plus `ug (ng, nc, nt)` (u at volume quadrature points), `qfd (ngf, nc, Dim, nf)`
(weighted gradient face fluxes), and `qtmp`/`q (npl, Dim, nc, nt)`
(pre-mass-solve and final LDG gradients). Allocated on the backend of `ctx`.
"""
struct RldgWorkspace{A3 <: AbstractArray{<:AbstractFloat, 3},
                     A4 <: AbstractArray{<:AbstractFloat, 4}}
    fng  :: A3
    fdg  :: A4
    srcg :: A3
    rtmp :: A3
    ug   :: A3
    qfd  :: A4
    qtmp :: A4
    q    :: A4
end

Adapt.@adapt_structure RldgWorkspace

function RldgWorkspace(ctx::DGContext{T}, nc::Integer) where {T}
    backend = KernelAbstractions.get_backend(ctx)
    dim = ndims(ctx)
    fng = KernelAbstractions.zeros(backend, T, ctx.ngf, nc, ctx.nf)
    fdg = KernelAbstractions.zeros(backend, T, ctx.ng, nc, dim, ctx.nt)
    srcg = KernelAbstractions.zeros(backend, T, ctx.ng, nc, ctx.nt)
    rtmp = KernelAbstractions.zeros(backend, T, ctx.npl, nc, ctx.nt)
    ug = KernelAbstractions.zeros(backend, T, ctx.ng, nc, ctx.nt)
    qfd = KernelAbstractions.zeros(backend, T, ctx.ngf, nc, dim, ctx.nf)
    qtmp = KernelAbstractions.zeros(backend, T, ctx.npl, dim, nc, ctx.nt)
    q = KernelAbstractions.zeros(backend, T, ctx.npl, dim, nc, ctx.nt)
    return RldgWorkspace(fng, fdg, srcg, rtmp, ug, qfd, qtmp, q)
end

"""
    getq!(q, ctx, phys, u, time; ws=RldgWorkspace(ctx, nvariables(phys)))

LDG gradient `q = ∇u`
(`(npl, Dim, nc, nt)`, already multiplied by the inverse mass matrix). Also
leaves `u` interpolated to volume quadrature points in `ws.ug` (reused by
[`rldgexpl!`](@ref)).
"""
function getq!(q, ctx::DGContext, phys::DGPhysics, u, time;
               ws::RldgWorkspace=RldgWorkspace(ctx, nvariables(phys)))
    backend = KernelAbstractions.get_backend(ctx)
    ncv = Val(nvariables(phys))
    dimv = Val(ndims(ctx))

    fill!(ws.qtmp, zero(eltype(ws.qtmp)))
    _interp_u!(backend)(ws.ug, u, ctx.shap, ncv; ndrange=(ctx.ng, ctx.nt))
    _grad_face_flux!(backend)(ws.qfd, u, ctx.facecon, ctx.f_el, ctx.nlg,
                              ctx.dws, ctx.pfg, ctx.shapf, phys.equation,
                              phys.boundary_conditions, time, ctx.ni, ncv, dimv;
                              ndrange=(ctx.ngf, ctx.nf))
    _grad_face_scatter!(backend)(ws.qtmp, ws.qfd, ctx.facecon, ctx.f_el,
                                 ctx.shapf, ctx.ni, ncv, dimv;
                                 ndrange=(ctx.npf, ctx.nf))
    _grad_volume!(backend)(ws.qtmp, ws.ug, ctx.shapd, ncv, dimv;
                           ndrange=(ctx.npl, ctx.nt))
    _grad_minv!(backend)(q, ws.qtmp, ctx.Minv, ncv, dimv; ndrange=(ctx.npl, ctx.nt))
    KernelAbstractions.synchronize(backend)

    return q
end

"""
    getq_ka(ctx, phys, u, time)

Allocating convenience wrapper around [`getq!`](@ref).
"""
function getq_ka(ctx::DGContext, phys::DGPhysics, u, time)
    backend = KernelAbstractions.get_backend(ctx)
    q = KernelAbstractions.zeros(backend, eltype(u), ctx.npl, ndims(ctx),
                                 nvariables(phys), ctx.nt)
    return getq!(q, ctx, phys, u, time)
end

"""
    rldgexpl!(r, ctx, phys, u, time; ws=RldgWorkspace(ctx, nvariables(phys)))

LDG (viscous) DG residual `r = du/dt` (already multiplied by the inverse mass
matrix) of the LDG discretization with viscous terms. Falls back to the
inviscid [`rinvexpl!`](@ref) kernels when the equation has no diffusion.
"""
function rldgexpl!(r, ctx::DGContext, phys::DGPhysics, u, time;
                   ws::RldgWorkspace=RldgWorkspace(ctx, nvariables(phys)))
    if !has_diffusion(phys)
        return rinvexpl!(r, ctx, phys, u, time; ws)
    end

    backend = KernelAbstractions.get_backend(ctx)
    ncv = Val(nvariables(phys))
    dimv = Val(ndims(ctx))
    eq = phys.equation

    getq!(ws.q, ctx, phys, u, time; ws)

    fill!(ws.rtmp, zero(eltype(ws.rtmp)))
    _face_flux_visc!(backend)(ws.fng, u, ws.q, ctx.facecon, ctx.f_el, ctx.nlg,
                              ctx.dws, ctx.pfg, ctx.shapf, eq, phys.numerical_flux,
                              phys.stabilization, phys.boundary_conditions,
                              time, ctx.ni, ncv, dimv; ndrange=(ctx.ngf, ctx.nf))
    _volume_flux_visc!(backend)(ws.fdg, ws.ug, ws.q, ctx.shap, ctx.pg,
                                eq, time, ncv, dimv; ndrange=(ctx.ng, ctx.nt))
    _face_scatter!(backend)(ws.rtmp, ws.fng, ctx.facecon, ctx.f_el, ctx.shapf,
                            ctx.ni, ncv; ndrange=(ctx.npf, ctx.nf))
    _volume_lift!(backend)(ws.rtmp, ws.fdg, ctx.shapd, ncv, dimv;
                           ndrange=(ctx.npl, ctx.nt))
    if phys.source !== nothing
        _volume_source!(backend)(ws.srcg, u, ctx.shap, ctx.pg, ctx.wjac,
                                 phys.source, time, ncv, dimv;
                                 ndrange=(ctx.ng, ctx.nt))
        _source_lift!(backend)(ws.rtmp, ws.srcg, ctx.shap, ncv;
                               ndrange=(ctx.npl, ctx.nt))
    end
    _apply_minv!(backend)(r, ws.rtmp, ctx.Minv, ncv; ndrange=(ctx.npl, ctx.nt))
    KernelAbstractions.synchronize(backend)

    return r
end

"""
    rldgexpl_ka(ctx, phys, u, time)

Allocating convenience wrapper around [`rldgexpl!`](@ref).
"""
rldgexpl_ka(ctx::DGContext, phys::DGPhysics, u, time) =
    rldgexpl!(similar(u), ctx, phys, u, time)

_default_ka_ws(::typeof(rldgexpl!), ctx, phys) = RldgWorkspace(ctx, nvariables(phys))
