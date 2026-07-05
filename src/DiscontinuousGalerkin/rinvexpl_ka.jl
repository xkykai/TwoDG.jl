# KernelAbstractions implementation of the inviscid DG residual. Works on the
# KA CPU backend and on any GPU backend (CUDA/AMDGPU/Metal/oneAPI) when the
# DGContext, workspace, state, and physics have been moved over with
# `Adapt.adapt` (e.g. `adapt(CuArray, ctx)`).
#
# Dimension-generic (THREED_PLAN Phase A): the same kernel bodies serve 2D
# triangles and 3D tetrahedra through the `Val(DIM)` parameter — positions and
# normals are `SVector{DIM}`, the volume flux is staged in one
# `fdg (ng, nc, Dim, nt)` array, and direction loops run `1:DIM`.
#
# Physics enters through dispatch on the `DGPhysics` components: the
# equation's `flux`, the problem's numerical flux, and `boundary_flux` on the
# per-tag boundary-condition tuple (statically selected, see
# Equations/boundary_conditions.jl).
#
# Kernel decomposition (staging buffers live in RinvWorkspace):
#   1. _face_flux!    (g, face):  interpolate uL/uR to face quad points, Riemann
#                                 or boundary flux, weight by dws  -> fng
#   2. _volume_flux!  (g, elem):  interpolate u to volume quad points, volume
#                                 flux                              -> fdg
#   3. _face_scatter! (j, face):  lift fng with shapf, atomic scatter into rtmp
#   4. _volume_lift!  (i, elem):  rtmp += Σ_d shapd_d*fdg_d (element-local)
#   5. optional source kernels
#   6. _apply_minv!   (i, elem):  r = Minv * rtmp

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using Atomix
using StaticArrays
using Adapt

@kernel function _face_flux!(fng, @Const(u), @Const(facecon), @Const(f_el),
                             @Const(nlg), @Const(dws), @Const(pfg), @Const(shapf),
                             eq, numflux, bcs, time, ni,
                             ::Val{NC}, ::Val{DIM}) where {NC, DIM}
    g, fc = @index(Global, NTuple)
    T = eltype(fng)
    npf = size(shapf, 1)

    el = f_el[fc, 1]
    n = SVector{DIM, T}(ntuple(d -> @inbounds(nlg[g, d, fc]), Val(DIM)))
    x = SVector{DIM, T}(ntuple(d -> @inbounds(pfg[g, d, fc]), Val(DIM)))

    uL = SVector{NC, T}(ntuple(Val(NC)) do c
        s = zero(T)
        @inbounds for j in 1:npf
            s += shapf[j, g] * u[facecon[j, 1, fc], c, el]
        end
        s
    end)

    if fc <= ni
        er = f_el[fc, 2]
        uR = SVector{NC, T}(ntuple(Val(NC)) do c
            s = zero(T)
            @inbounds for j in 1:npf
                s += shapf[j, g] * u[facecon[j, 2, fc], c, er]
            end
            s
        end)
        fn = numflux(eq, uL, uR, n, x, time)
    else
        ib = -Int(f_el[fc, 2])
        fn = apply_boundary_flux(bcs, ib, eq, numflux, uL, n, x, time)
    end

    w = dws[g, fc]
    @inbounds for c in 1:NC
        fng[g, c, fc] = fn[c] * w
    end
end

@kernel function _face_scatter!(rtmp, @Const(fng), @Const(facecon), @Const(f_el),
                                @Const(shapf), ni, ::Val{NC}) where {NC}
    j, fc = @index(Global, NTuple)
    T = eltype(rtmp)
    ngf = size(shapf, 2)

    el = f_el[fc, 1]
    il = facecon[j, 1, fc]
    interior = fc <= ni

    @inbounds for c in 1:NC
        cnt = zero(T)
        for g in 1:ngf
            cnt += shapf[j, g] * fng[g, c, fc]
        end
        Atomix.@atomic rtmp[il, c, el] -= cnt
        if interior
            er = f_el[fc, 2]
            ir = facecon[j, 2, fc]
            Atomix.@atomic rtmp[ir, c, er] += cnt
        end
    end
end

@kernel function _volume_flux!(fdg, @Const(u), @Const(shap), @Const(pg),
                               eq, time, ::Val{NC}, ::Val{DIM}) where {NC, DIM}
    g, e = @index(Global, NTuple)
    T = eltype(fdg)
    npl = size(shap, 1)

    ug = SVector{NC, T}(ntuple(Val(NC)) do c
        s = zero(T)
        @inbounds for i in 1:npl
            s += shap[i, g] * u[i, c, e]
        end
        s
    end)
    x = SVector{DIM, T}(ntuple(d -> @inbounds(pg[g, d, e]), Val(DIM)))

    fd = flux(eq, ug, x, time)
    @inbounds for d in 1:DIM, c in 1:NC
        fdg[g, c, d, e] = fd[d][c]
    end
end

@kernel function _volume_lift!(rtmp, @Const(fdg), @Const(shapd),
                               ::Val{NC}, ::Val{DIM}) where {NC, DIM}
    i, e = @index(Global, NTuple)
    T = eltype(rtmp)
    ng = size(fdg, 1)

    @inbounds for c in 1:NC
        acc = zero(T)
        for g in 1:ng
            s = zero(T)
            for d in 1:DIM
                s += shapd[i, g, d, e] * fdg[g, c, d, e]
            end
            acc += s
        end
        rtmp[i, c, e] += acc
    end
end

@kernel function _volume_source!(srcg, @Const(u), @Const(shap), @Const(pg),
                                 @Const(wjac), src, time,
                                 ::Val{NC}, ::Val{DIM}) where {NC, DIM}
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
    x = SVector{DIM, T}(ntuple(d -> @inbounds(pg[g, d, e]), Val(DIM)))

    sv = src(ug, x, time)
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
`fng (ngf, nc, nf)` weighted face fluxes, `fdg (ng, nc, Dim, nt)` volume
fluxes, `srcg (ng, nc, nt)` source, `rtmp (npl, nc, nt)` pre-mass-solve
residual. Allocated on the backend of `ctx`.
"""
struct RinvWorkspace{A3 <: AbstractArray{<:AbstractFloat, 3},
                     A4 <: AbstractArray{<:AbstractFloat, 4}}
    fng  :: A3
    fdg  :: A4
    srcg :: A3
    rtmp :: A3
end

Adapt.@adapt_structure RinvWorkspace

function RinvWorkspace(ctx::DGContext{T}, nc::Integer) where {T}
    backend = KernelAbstractions.get_backend(ctx)
    dim = ndims(ctx)
    fng = KernelAbstractions.zeros(backend, T, ctx.ngf, nc, ctx.nf)
    fdg = KernelAbstractions.zeros(backend, T, ctx.ng, nc, dim, ctx.nt)
    srcg = KernelAbstractions.zeros(backend, T, ctx.ng, nc, ctx.nt)
    rtmp = KernelAbstractions.zeros(backend, T, ctx.npl, nc, ctx.nt)
    return RinvWorkspace(fng, fdg, srcg, rtmp)
end

"""
    rinvexpl!(r, ctx, phys::DGPhysics, u, time; ws=RinvWorkspace(ctx, nvariables(phys)))

Inviscid DG residual `r = du/dt` (already multiplied by the inverse mass
matrix). `r`, `u` are `(npl, nc, nt)` arrays on the same backend as `ctx`.
Pass a pre-built `ws` to avoid re-allocating staging buffers across calls.
"""
function rinvexpl!(r, ctx::DGContext, phys::DGPhysics, u, time;
                   ws=RinvWorkspace(ctx, nvariables(phys)))
    backend = KernelAbstractions.get_backend(ctx)
    ncv = Val(nvariables(phys))
    dimv = Val(ndims(ctx))
    eq = phys.equation

    # Kernels launched on the same backend execute in order (same stream/queue
    # on GPUs; synchronous on the CPU backend), so no intermediate syncs needed.
    fill!(ws.rtmp, zero(eltype(ws.rtmp)))
    _face_flux!(backend)(ws.fng, u, ctx.facecon, ctx.f_el, ctx.nlg, ctx.dws,
                         ctx.pfg, ctx.shapf, eq, phys.numerical_flux,
                         phys.boundary_conditions, time, ctx.ni, ncv, dimv;
                         ndrange=(ctx.ngf, ctx.nf))
    _volume_flux!(backend)(ws.fdg, u, ctx.shap, ctx.pg, eq, time, ncv, dimv;
                           ndrange=(ctx.ng, ctx.nt))
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
    rinvexpl_ka(ctx, phys, u, time)

Allocating convenience wrapper around [`rinvexpl!`](@ref).
"""
rinvexpl_ka(ctx::DGContext, phys::DGPhysics, u, time) =
    rinvexpl!(similar(u), ctx, phys, u, time)

"""
    rk4_ka!([residual!,] ctx, phys, u, time, dt, nstep; ws)

In-place RK4 time integrator driving a KA residual (`rinvexpl!` by default;
pass `rldgexpl!` for the LDG viscous path). All stage buffers are allocated
once up front; the stage updates are plain broadcasts, so the stepper runs
unchanged on CPU and GPU arrays. Returns `u` after `nstep` steps.
"""
function rk4_ka!(residual!::F, ctx::DGContext, phys::DGPhysics, u, time::Real,
                 dt::Real, nstep::Integer;
                 ws=_default_ka_ws(residual!, ctx, phys)) where {F}
    T = eltype(u)
    k1, k2, k3, k4, tmp = (similar(u) for _ in 1:5)
    h = T(dt)
    t = time

    for _ in 1:nstep
        residual!(k1, ctx, phys, u, t; ws)
        @. tmp = u + h / 2 * k1
        residual!(k2, ctx, phys, tmp, t + dt / 2; ws)
        @. tmp = u + h / 2 * k2
        residual!(k3, ctx, phys, tmp, t + dt / 2; ws)
        @. tmp = u + h * k3
        residual!(k4, ctx, phys, tmp, t + dt; ws)
        @. u += h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
        t += dt
    end

    return u
end

rk4_ka!(ctx::DGContext, phys::DGPhysics, u, time::Real, dt::Real, nstep::Integer;
        ws=RinvWorkspace(ctx, nvariables(phys))) =
    rk4_ka!(rinvexpl!, ctx, phys, u, time, dt, nstep; ws)

_default_ka_ws(::typeof(rinvexpl!), ctx, phys) = RinvWorkspace(ctx, nvariables(phys))
