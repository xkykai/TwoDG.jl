# KernelAbstractions implementation of the LDG viscous path (gradient
# computation + viscous fluxes). Shares DGContext and the scatter/lift/mass
# kernels with rinvexpl_ka.jl.
#
# Physics enters through dispatch on the `DGPhysics` components: the
# equation's `viscous_flux`, the stabilization policy's
# `viscous_numerical_flux`, and `boundary_trace`/`boundary_viscous_flux` on
# the per-tag boundary-condition tuple. Gradients q are SMatrix{2, nc}
# (rows = x/y derivative).
#
# LDG alternating traces: û on interior faces is the LEFT trace, and the
# viscous interface flux sees the RIGHT element's gradient. Gradient layout:
# q (npl, 2, nc, nt).

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using Atomix
using StaticArrays
using Adapt

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
@inline function _face_state(u, facecon, sh1d, g, fc, side, e, ::Val{NC}) where {NC}
    T = eltype(u)
    np1d = size(sh1d, 1)
    return SVector{NC, T}(ntuple(Val(NC)) do c
        s = zero(T)
        @inbounds for j in 1:np1d
            s += sh1d[j, g] * u[facecon[j, side, fc], c, e]
        end
        s
    end)
end

@inline function _face_grad(q, facecon, sh1d, g, fc, side, e, ::Val{NC}) where {NC}
    T = eltype(q)
    np1d = size(sh1d, 1)
    return SMatrix{2, NC, T}(ntuple(Val(2 * NC)) do k
        d = (k - 1) % 2 + 1
        c = (k - 1) ÷ 2 + 1
        s = zero(T)
        @inbounds for j in 1:np1d
            s += sh1d[j, g] * q[facecon[j, side, fc], d, c, e]
        end
        s
    end)
end

# LDG gradient, face term: û (left trace on interior faces, boundary_trace on
# the boundary) times weighted normal components -> qfx, qfy (ng1d, nc, nf).
@kernel function _grad_face_flux!(qfx, qfy, @Const(u), @Const(facecon), @Const(f_el),
                                  @Const(nlg), @Const(dws), @Const(pfg), @Const(sh1d),
                                  eq, bcs, time, ni, ::Val{NC}) where {NC}
    g, fc = @index(Global, NTuple)

    el = f_el[fc, 1]
    uL = _face_state(u, facecon, sh1d, g, fc, 1, el, Val(NC))

    if fc <= ni
        û = uL
    else
        n = SVector(nlg[g, 1, fc], nlg[g, 2, fc])
        x = SVector(pfg[g, 1, fc], pfg[g, 2, fc])
        ib = -Int(f_el[fc, 2])
        û = apply_boundary_trace(bcs, ib, eq, uL, n, x, time)
    end

    w = dws[g, fc]
    @inbounds for c in 1:NC
        qfx[g, c, fc] = û[c] * w * nlg[g, 1, fc]
        qfy[g, c, fc] = û[c] * w * nlg[g, 2, fc]
    end
end

# lift the face term with sh1d and scatter into qtmp (+ left, - right)
@kernel function _grad_face_scatter!(qtmp, @Const(qfx), @Const(qfy), @Const(facecon),
                                     @Const(f_el), @Const(sh1d), ni, ::Val{NC}) where {NC}
    j, fc = @index(Global, NTuple)
    T = eltype(qtmp)
    ng1d = size(sh1d, 2)

    el = f_el[fc, 1]
    il = facecon[j, 1, fc]
    interior = fc <= ni

    @inbounds for c in 1:NC
        cx = zero(T)
        cy = zero(T)
        for g in 1:ng1d
            cx += sh1d[j, g] * qfx[g, c, fc]
            cy += sh1d[j, g] * qfy[g, c, fc]
        end
        Atomix.@atomic qtmp[il, 1, c, el] += cx
        Atomix.@atomic qtmp[il, 2, c, el] += cy
        if interior
            er = f_el[fc, 2]
            ir = facecon[j, 2, fc]
            Atomix.@atomic qtmp[ir, 1, c, er] -= cx
            Atomix.@atomic qtmp[ir, 2, c, er] -= cy
        end
    end
end

# LDG gradient, volume term: qtmp -= ∫ u ∇φ (element-local, no atomics)
@kernel function _grad_volume!(qtmp, @Const(ug), @Const(shapx), @Const(shapy),
                               ::Val{NC}) where {NC}
    i, e = @index(Global, NTuple)
    T = eltype(qtmp)
    ng = size(ug, 1)

    @inbounds for c in 1:NC
        ax = zero(T)
        ay = zero(T)
        for g in 1:ng
            ax += shapx[i, g, e] * ug[g, c, e]
            ay += shapy[i, g, e] * ug[g, c, e]
        end
        qtmp[i, 1, c, e] -= ax
        qtmp[i, 2, c, e] -= ay
    end
end

@kernel function _grad_minv!(q, @Const(qtmp), @Const(Minv), ::Val{NC}) where {NC}
    i, e = @index(Global, NTuple)
    T = eltype(q)
    npl = size(Minv, 1)

    @inbounds for c in 1:NC, d in 1:2
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
                                  @Const(sh1d), eq, numflux, stab, bcs,
                                  time, ni, ::Val{NC}) where {NC}
    g, fc = @index(Global, NTuple)

    el = f_el[fc, 1]
    n = SVector(nlg[g, 1, fc], nlg[g, 2, fc])
    x = SVector(pfg[g, 1, fc], pfg[g, 2, fc])

    uL = _face_state(u, facecon, sh1d, g, fc, 1, el, Val(NC))

    if fc <= ni
        er = f_el[fc, 2]
        uR = _face_state(u, facecon, sh1d, g, fc, 2, er, Val(NC))
        qL = _face_grad(q, facecon, sh1d, g, fc, 1, el, Val(NC))
        qR = _face_grad(q, facecon, sh1d, g, fc, 2, er, Val(NC))
        fn = numflux(eq, uL, uR, n, x, time) +
             viscous_numerical_flux(stab, eq, uL, uR, qL, qR, n, x, time)
    else
        ib = -Int(f_el[fc, 2])
        qL = _face_grad(q, facecon, sh1d, g, fc, 1, el, Val(NC))
        fn = apply_boundary_flux(bcs, ib, eq, numflux, uL, n, x, time) +
             apply_boundary_viscous_flux(bcs, ib, stab, eq, uL, qL, n, x, time)
    end

    w = dws[g, fc]
    @inbounds for c in 1:NC
        fng[g, c, fc] = fn[c] * w
    end
end

# volume flux with viscous contribution (reads the staged ug)
@kernel function _volume_flux_visc!(fxg, fyg, @Const(ug), @Const(q), @Const(shap),
                                    @Const(pg), eq, time, ::Val{NC}) where {NC}
    g, e = @index(Global, NTuple)
    T = eltype(fxg)
    npl = size(shap, 1)

    u = SVector{NC, T}(ntuple(c -> @inbounds(ug[g, c, e]), Val(NC)))
    qg = SMatrix{2, NC, T}(ntuple(Val(2 * NC)) do k
        d = (k - 1) % 2 + 1
        c = (k - 1) ÷ 2 + 1
        s = zero(T)
        @inbounds for i in 1:npl
            s += shap[i, g] * q[i, d, c, e]
        end
        s
    end)
    x = SVector(pg[g, 1, e], pg[g, 2, e])

    fxi, fyi = flux(eq, u, x, time)
    fxv, fyv = viscous_flux(eq, u, qg, x, time)
    @inbounds for c in 1:NC
        fxg[g, c, e] = fxi[c] + fxv[c]
        fyg[g, c, e] = fyi[c] + fyv[c]
    end
end

"""
    RldgWorkspace(ctx, nc)

Staging buffers for the LDG residual path: everything in [`RinvWorkspace`](@ref)
plus `ug (ng, nc, nt)` (u at volume quadrature points), `qfx`/`qfy
(ng1d, nc, nf)` (weighted gradient face fluxes), and `qtmp`/`q (npl, 2, nc, nt)`
(pre-mass-solve and final LDG gradients). Allocated on the backend of `ctx`.
"""
struct RldgWorkspace{A3 <: AbstractArray{<:AbstractFloat, 3},
                     A4 <: AbstractArray{<:AbstractFloat, 4}}
    fng  :: A3
    fxg  :: A3
    fyg  :: A3
    srcg :: A3
    rtmp :: A3
    ug   :: A3
    qfx  :: A3
    qfy  :: A3
    qtmp :: A4
    q    :: A4
end

Adapt.@adapt_structure RldgWorkspace

function RldgWorkspace(ctx::DGContext{T}, nc::Integer) where {T}
    backend = KernelAbstractions.get_backend(ctx)
    fng = KernelAbstractions.zeros(backend, T, ctx.ng1d, nc, ctx.nf)
    fxg = KernelAbstractions.zeros(backend, T, ctx.ng, nc, ctx.nt)
    fyg = KernelAbstractions.zeros(backend, T, ctx.ng, nc, ctx.nt)
    srcg = KernelAbstractions.zeros(backend, T, ctx.ng, nc, ctx.nt)
    rtmp = KernelAbstractions.zeros(backend, T, ctx.npl, nc, ctx.nt)
    ug = KernelAbstractions.zeros(backend, T, ctx.ng, nc, ctx.nt)
    qfx = KernelAbstractions.zeros(backend, T, ctx.ng1d, nc, ctx.nf)
    qfy = KernelAbstractions.zeros(backend, T, ctx.ng1d, nc, ctx.nf)
    qtmp = KernelAbstractions.zeros(backend, T, ctx.npl, 2, nc, ctx.nt)
    q = KernelAbstractions.zeros(backend, T, ctx.npl, 2, nc, ctx.nt)
    return RldgWorkspace(fng, fxg, fyg, srcg, rtmp, ug, qfx, qfy, qtmp, q)
end

"""
    getq!(q, ctx, phys, u, time; ws=RldgWorkspace(ctx, nvariables(phys)))

LDG gradient `q = ∇u`
(`(npl, 2, nc, nt)`, already multiplied by the inverse mass matrix). Also
leaves `u` interpolated to volume quadrature points in `ws.ug` (reused by
[`rldgexpl!`](@ref)).
"""
function getq!(q, ctx::DGContext, phys::DGPhysics, u, time;
               ws::RldgWorkspace=RldgWorkspace(ctx, nvariables(phys)))
    backend = KernelAbstractions.get_backend(ctx)
    ncv = Val(nvariables(phys))

    fill!(ws.qtmp, zero(eltype(ws.qtmp)))
    _interp_u!(backend)(ws.ug, u, ctx.shap, ncv; ndrange=(ctx.ng, ctx.nt))
    _grad_face_flux!(backend)(ws.qfx, ws.qfy, u, ctx.facecon, ctx.f_el, ctx.nlg,
                              ctx.dws, ctx.pfg, ctx.sh1d, phys.equation,
                              phys.boundary_conditions, time, ctx.ni, ncv;
                              ndrange=(ctx.ng1d, ctx.nf))
    _grad_face_scatter!(backend)(ws.qtmp, ws.qfx, ws.qfy, ctx.facecon, ctx.f_el,
                                 ctx.sh1d, ctx.ni, ncv; ndrange=(ctx.np1d, ctx.nf))
    _grad_volume!(backend)(ws.qtmp, ws.ug, ctx.shapx, ctx.shapy, ncv;
                           ndrange=(ctx.npl, ctx.nt))
    _grad_minv!(backend)(q, ws.qtmp, ctx.Minv, ncv; ndrange=(ctx.npl, ctx.nt))
    KernelAbstractions.synchronize(backend)

    return q
end

"""
    getq_ka(ctx, phys, u, time)

Allocating convenience wrapper around [`getq!`](@ref).
"""
function getq_ka(ctx::DGContext, phys::DGPhysics, u, time)
    backend = KernelAbstractions.get_backend(ctx)
    q = KernelAbstractions.zeros(backend, eltype(u), ctx.npl, 2, nvariables(phys), ctx.nt)
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
    eq = phys.equation

    getq!(ws.q, ctx, phys, u, time; ws)

    fill!(ws.rtmp, zero(eltype(ws.rtmp)))
    _face_flux_visc!(backend)(ws.fng, u, ws.q, ctx.facecon, ctx.f_el, ctx.nlg,
                              ctx.dws, ctx.pfg, ctx.sh1d, eq, phys.numerical_flux,
                              phys.stabilization, phys.boundary_conditions,
                              time, ctx.ni, ncv; ndrange=(ctx.ng1d, ctx.nf))
    _volume_flux_visc!(backend)(ws.fxg, ws.fyg, ws.ug, ws.q, ctx.shap, ctx.pg,
                                eq, time, ncv; ndrange=(ctx.ng, ctx.nt))
    _face_scatter!(backend)(ws.rtmp, ws.fng, ctx.facecon, ctx.f_el, ctx.sh1d,
                            ctx.ni, ncv; ndrange=(ctx.np1d, ctx.nf))
    _volume_lift!(backend)(ws.rtmp, ws.fxg, ws.fyg, ctx.shapx, ctx.shapy, ncv;
                           ndrange=(ctx.npl, ctx.nt))
    if phys.source !== nothing
        _volume_source!(backend)(ws.srcg, u, ctx.shap, ctx.pg, ctx.wjac,
                                 phys.source, time, ncv;
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
