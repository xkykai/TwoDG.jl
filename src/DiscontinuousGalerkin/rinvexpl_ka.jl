# KernelAbstractions implementation of the inviscid DG residual (the GPU-capable
# counterpart of `rinvexpl`). Works on the KA CPU backend and on any GPU backend
# (CUDA/AMDGPU/Metal/oneAPI) when the DGContext, workspace, state and app have
# been moved over with `Adapt.adapt` (e.g. `adapt(CuArray, ctx)`).
#
# Requires an app in the *pointwise* flux convention (see src/Apps/pointwise.jl
# and GPU_PLAN.md); legacy matrix-flux apps will not work here.
#
# Kernel decomposition (staging buffers live in RinvWorkspace):
#   1. _face_flux!    (g, face):  interpolate uL/uR to face quad points, Riemann
#                                 or boundary flux, weight by dws  -> fng
#   2. _volume_flux!  (g, elem):  interpolate u to volume quad points, volume
#                                 flux                              -> fxg, fyg
#   3. _face_scatter! (j, face):  lift fng with sh1d, atomic scatter into rtmp
#                                 (atomics make this race-free, unlike the
#                                 threaded legacy loop)
#   4. _volume_lift!  (i, elem):  rtmp += shapx*fxg + shapy*fyg (element-local)
#   5. optional source kernels
#   6. _apply_minv!   (i, elem):  r = Minv * rtmp

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using Atomix
using StaticArrays
using Adapt

@kernel function _face_flux!(fng, @Const(u), @Const(facecon), @Const(f_el),
                             @Const(nlg), @Const(dws), @Const(pfg), @Const(sh1d),
                             finvi, finvb, @Const(bcm), @Const(bcs),
                             param, time, ni, ::Val{NC}) where {NC}
    g, fc = @index(Global, NTuple)
    T = eltype(fng)
    np1d = size(sh1d, 1)

    el = f_el[fc, 1]
    n = SVector(nlg[g, 1, fc], nlg[g, 2, fc])
    x = SVector(pfg[g, 1, fc], pfg[g, 2, fc])

    uL = SVector{NC, T}(ntuple(Val(NC)) do c
        s = zero(T)
        @inbounds for j in 1:np1d
            s += sh1d[j, g] * u[facecon[j, 1, fc], c, el]
        end
        s
    end)

    if fc <= ni
        er = f_el[fc, 2]
        uR = SVector{NC, T}(ntuple(Val(NC)) do c
            s = zero(T)
            @inbounds for j in 1:np1d
                s += sh1d[j, g] * u[facecon[j, 2, fc], c, er]
            end
            s
        end)
        fn = finvi(uL, uR, n, x, param, time)
    else
        ib = -Int(f_el[fc, 2])
        ibc = bcm[ib]
        ui = SVector{NC, T}(ntuple(c -> T(bcs[ibc, c]), Val(NC)))
        fn = finvb(uL, n, ibc, ui, x, param, time)
    end

    w = dws[g, fc]
    @inbounds for c in 1:NC
        fng[g, c, fc] = fn[c] * w
    end
end

@kernel function _face_scatter!(rtmp, @Const(fng), @Const(facecon), @Const(f_el),
                                @Const(sh1d), ni, ::Val{NC}) where {NC}
    j, fc = @index(Global, NTuple)
    T = eltype(rtmp)
    ng1d = size(sh1d, 2)

    el = f_el[fc, 1]
    il = facecon[j, 1, fc]
    interior = fc <= ni

    @inbounds for c in 1:NC
        cnt = zero(T)
        for g in 1:ng1d
            cnt += sh1d[j, g] * fng[g, c, fc]
        end
        Atomix.@atomic rtmp[il, c, el] -= cnt
        if interior
            er = f_el[fc, 2]
            ir = facecon[j, 2, fc]
            Atomix.@atomic rtmp[ir, c, er] += cnt
        end
    end
end

@kernel function _volume_flux!(fxg, fyg, @Const(u), @Const(shap), @Const(pg),
                               finvv, param, time, ::Val{NC}) where {NC}
    g, e = @index(Global, NTuple)
    T = eltype(fxg)
    npl = size(shap, 1)

    ug = SVector{NC, T}(ntuple(Val(NC)) do c
        s = zero(T)
        @inbounds for i in 1:npl
            s += shap[i, g] * u[i, c, e]
        end
        s
    end)
    x = SVector(pg[g, 1, e], pg[g, 2, e])

    fx, fy = finvv(ug, x, param, time)
    @inbounds for c in 1:NC
        fxg[g, c, e] = fx[c]
        fyg[g, c, e] = fy[c]
    end
end

@kernel function _volume_lift!(rtmp, @Const(fxg), @Const(fyg),
                               @Const(shapx), @Const(shapy), ::Val{NC}) where {NC}
    i, e = @index(Global, NTuple)
    T = eltype(rtmp)
    ng = size(fxg, 1)

    @inbounds for c in 1:NC
        acc = zero(T)
        for g in 1:ng
            acc += shapx[i, g, e] * fxg[g, c, e] + shapy[i, g, e] * fyg[g, c, e]
        end
        rtmp[i, c, e] += acc
    end
end

@kernel function _volume_source!(srcg, @Const(u), @Const(shap), @Const(pg),
                                 @Const(wjac), src, param, time, ::Val{NC}) where {NC}
    g, e = @index(Global, NTuple)
    T = eltype(srcg)
    npl = size(shap, 1)

    ug = SVector{NC, T}(ntuple(Val(NC)) do c
        s = zero(T)
        @inbounds for i in 1:npl
            s += shap[i, g] * u[i, c, e]
        end
        s
    end)
    x = SVector(pg[g, 1, e], pg[g, 2, e])

    sv = src(ug, x, param, time)
    w = wjac[g, e]
    @inbounds for c in 1:NC
        srcg[g, c, e] = sv[c] * w
    end
end

@kernel function _source_lift!(rtmp, @Const(srcg), @Const(shap), ::Val{NC}) where {NC}
    i, e = @index(Global, NTuple)
    T = eltype(rtmp)
    ng = size(srcg, 1)

    @inbounds for c in 1:NC
        acc = zero(T)
        for g in 1:ng
            acc += shap[i, g] * srcg[g, c, e]
        end
        rtmp[i, c, e] += acc
    end
end

@kernel function _apply_minv!(r, @Const(rtmp), @Const(Minv), ::Val{NC}) where {NC}
    i, e = @index(Global, NTuple)
    T = eltype(r)
    npl = size(Minv, 1)

    @inbounds for c in 1:NC
        acc = zero(T)
        for j in 1:npl
            acc += Minv[i, j, e] * rtmp[j, c, e]
        end
        r[i, c, e] = acc
    end
end

"""
    RinvWorkspace(ctx, nc)

Staging buffers reused across residual evaluations (e.g. RK stages):
`fng (ng1d, nc, nf)` weighted face fluxes, `fxg`/`fyg`/`srcg (ng, nc, nt)`
volume fluxes and source, `rtmp (npl, nc, nt)` pre-mass-solve residual.
Allocated on the backend of `ctx`.
"""
struct RinvWorkspace{A3 <: AbstractArray{<:AbstractFloat, 3}}
    fng  :: A3
    fxg  :: A3
    fyg  :: A3
    srcg :: A3
    rtmp :: A3
end

Adapt.@adapt_structure RinvWorkspace

function RinvWorkspace(ctx::DGContext{T}, nc::Integer) where {T}
    backend = KernelAbstractions.get_backend(ctx)
    fng = KernelAbstractions.zeros(backend, T, ctx.ng1d, nc, ctx.nf)
    fxg = KernelAbstractions.zeros(backend, T, ctx.ng, nc, ctx.nt)
    fyg = KernelAbstractions.zeros(backend, T, ctx.ng, nc, ctx.nt)
    srcg = KernelAbstractions.zeros(backend, T, ctx.ng, nc, ctx.nt)
    rtmp = KernelAbstractions.zeros(backend, T, ctx.npl, nc, ctx.nt)
    return RinvWorkspace(fng, fxg, fyg, srcg, rtmp)
end

"""
    rinvexpl!(r, ctx, app, u, time; ws=RinvWorkspace(ctx, app.nc))

KernelAbstractions version of [`rinvexpl`](@ref): residual `r = du/dt` (already
multiplied by the inverse mass matrix) of the inviscid DG discretization, for
an `app` in the pointwise flux convention. `r`, `u` are `(npl, nc, nt)` arrays
on the same backend as `ctx`. Pass a pre-built `ws` to avoid re-allocating
staging buffers across calls.
"""
function rinvexpl!(r, ctx::DGContext, app, u, time;
                   ws=RinvWorkspace(ctx, app.nc))
    backend = KernelAbstractions.get_backend(ctx)
    ncv = Val(Int(app.nc))

    # Kernels launched on the same backend execute in order (same stream/queue
    # on GPUs; synchronous on the CPU backend), so no intermediate syncs needed.
    fill!(ws.rtmp, zero(eltype(ws.rtmp)))
    _face_flux!(backend)(ws.fng, u, ctx.facecon, ctx.f_el, ctx.nlg, ctx.dws,
                         ctx.pfg, ctx.sh1d, app.finvi, app.finvb, app.bcm,
                         app.bcs, app.arg, time, ctx.ni, ncv;
                         ndrange=(ctx.ng1d, ctx.nf))
    _volume_flux!(backend)(ws.fxg, ws.fyg, u, ctx.shap, ctx.pg, app.finvv,
                           app.arg, time, ncv; ndrange=(ctx.ng, ctx.nt))
    _face_scatter!(backend)(ws.rtmp, ws.fng, ctx.facecon, ctx.f_el, ctx.sh1d,
                            ctx.ni, ncv; ndrange=(ctx.np1d, ctx.nf))
    _volume_lift!(backend)(ws.rtmp, ws.fxg, ws.fyg, ctx.shapx, ctx.shapy, ncv;
                           ndrange=(ctx.npl, ctx.nt))
    if app.src !== nothing
        _volume_source!(backend)(ws.srcg, u, ctx.shap, ctx.pg, ctx.wjac,
                                 app.src, app.arg, time, ncv;
                                 ndrange=(ctx.ng, ctx.nt))
        _source_lift!(backend)(ws.rtmp, ws.srcg, ctx.shap, ncv;
                               ndrange=(ctx.npl, ctx.nt))
    end
    _apply_minv!(backend)(r, ws.rtmp, ctx.Minv, ncv; ndrange=(ctx.npl, ctx.nt))
    KernelAbstractions.synchronize(backend)

    return r
end

"""
    rinvexpl_ka(ctx, app, u, time)

Allocating convenience wrapper around [`rinvexpl!`](@ref).
"""
rinvexpl_ka(ctx::DGContext, app, u, time) = rinvexpl!(similar(u), ctx, app, u, time)

"""
    rk4_ka!([residual!,] ctx, app, u, time, dt, nstep; ws)

In-place RK4 time integrator driving a KA residual (`rinvexpl!` by default;
pass `rldgexpl!` for the LDG viscous path). All stage buffers are allocated
once up front; the stage updates are plain broadcasts, so the stepper runs
unchanged on CPU and GPU arrays. Returns `u` after `nstep` steps.
"""
function rk4_ka!(residual!::F, ctx::DGContext, app, u, time::Real, dt::Real,
                 nstep::Integer; ws=_default_ka_ws(residual!, ctx, app)) where {F}
    T = eltype(u)
    k1, k2, k3, k4, tmp = (similar(u) for _ in 1:5)
    h = T(dt)
    t = time

    for _ in 1:nstep
        residual!(k1, ctx, app, u, t; ws)
        @. tmp = u + h / 2 * k1
        residual!(k2, ctx, app, tmp, t + dt / 2; ws)
        @. tmp = u + h / 2 * k2
        residual!(k3, ctx, app, tmp, t + dt / 2; ws)
        @. tmp = u + h * k3
        residual!(k4, ctx, app, tmp, t + dt; ws)
        @. u += h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
        t += dt
    end

    return u
end

rk4_ka!(ctx::DGContext, app, u, time::Real, dt::Real, nstep::Integer;
        ws=RinvWorkspace(ctx, app.nc)) =
    rk4_ka!(rinvexpl!, ctx, app, u, time, dt, nstep; ws)

_default_ka_ws(::typeof(rinvexpl!), ctx, app) = RinvWorkspace(ctx, app.nc)
