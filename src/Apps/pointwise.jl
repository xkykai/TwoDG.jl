# Pointwise (per-quadrature-point) flux functions for the GPU-capable
# KernelAbstractions residual path (`rinvexpl!`). Unlike the legacy whole-face
# matrix fluxes, these are allocation-free scalar functions over SVector
# states with NamedTuple parameters, so they compile inside GPU kernels.
#
# Calling convention (see GPU_PLAN.md):
#   finvi(uL, uR, n, x, param, time) -> SVector{nc}   interface normal flux
#   finvb(uL, n, ib, ui, x, param, time) -> SVector{nc}   boundary normal flux
#   finvv(u, x, param, time) -> (fx, fy)::NTuple{2,SVector{nc}}   volume flux
#   src(u, x, param, time) -> SVector{nc}   source (optional)
#
# Viscous (LDG) convention — gradients are SMatrix{2,nc} (rows = x/y derivative):
#   fvisi(uL, uR, qL, qR, n, x, param, time) -> SVector{nc}   interface flux
#   fvisb(uL, qL, n, ib, ui, x, param, time) -> SVector{nc}   boundary flux
#   fvisv(u, q, x, param, time) -> (fx, fy)                   volume flux
#   fvisub(uL, n, ib, ui, x, param, time) -> SVector{nc}      boundary trace û
#
# `param` must be a NamedTuple (GPU kernels cannot use Dict). The `mkapp_*_pt`
# constructors return a regular `App` whose callbacks follow this convention;
# such apps work only with the KA path, not with the legacy `rinvexpl`.

using StaticArrays
using Adapt

Adapt.@adapt_structure App

# ------------------------------------------------------------------ convection

# Velocity field: either a constant SVector{2} or a callable x -> SVector{2}.
@inline _convection_velocity(vf::SVector{2}, x) = vf
@inline _convection_velocity(vf, x) = vf(x)

@inline function convectioni_pt(uL::SVector{1, T}, uR::SVector{1, T}, n::SVector{2, T},
                                x::SVector{2, T}, param, time) where {T}
    v = _convection_velocity(param.vf, x)
    vn = v[1] * n[1] + v[2] * n[2]
    return T(0.5) * vn * (uL + uR) + T(0.5) * abs(vn) * (uL - uR)
end

@inline convectionb_pt(uL::SVector{1}, n, ib, ui, x, param, time) =
    convectioni_pt(uL, ui, n, x, param, time)

@inline function convectionv_pt(u::SVector{1, T}, x::SVector{2, T}, param, time) where {T}
    v = _convection_velocity(param.vf, x)
    return T(v[1]) * u, T(v[2]) * u
end

"""
    mkapp_convection_pt(vf; bcm=nothing, bcs=nothing, src=nothing)

Linear convection app in the pointwise (GPU-capable) convention. `vf` is a
constant `SVector{2}` velocity or a callable `x::SVector{2} -> SVector{2}`.
"""
mkapp_convection_pt(vf; bcm=nothing, bcs=nothing, src=nothing) =
    App(; nc=1, pg=true, arg=(; vf), bcm, bcs, src,
        finvi=convectioni_pt, finvb=convectionb_pt, finvv=convectionv_pt)

# ------------------------------------------------------------------------ wave

@inline function wavei_roe_pt(uL::SVector{3, T}, uR::SVector{3, T}, n::SVector{2, T},
                              x::SVector{2, T}, param, time) where {T}
    c = T(param.c)
    ca = abs(c)
    n1, n2 = n
    # central part: f_x(u) = -c (u3, 0, u1), f_y(u) = -c (0, u3, u2)
    fav1 = -c * T(0.5) * n1 * (uL[3] + uR[3])
    fav2 = -c * T(0.5) * n2 * (uL[3] + uR[3])
    fav3 = -c * T(0.5) * (n1 * (uL[1] + uR[1]) + n2 * (uL[2] + uR[2]))
    # upwind part
    qb = T(0.5) * ca * ((uL[1] - uR[1]) * n1 + (uL[2] - uR[2]) * n2)
    ub = T(0.5) * ca * (uL[3] - uR[3])
    return SVector(fav1 + qb * n1, fav2 + qb * n2, fav3 + ub)
end

@inline function waveb_pt(uL::SVector{3, T}, n::SVector{2, T}, ib, ui::SVector{3, T},
                          x::SVector{2, T}, param, time) where {T}
    if ib == 2      # solid wall (reflection)
        un = uL[1] * n[1] + uL[2] * n[2]
        uR = SVector(uL[1] - 2un * n[1], uL[2] - 2un * n[2], uL[3])
    elseif ib == 3  # incoming wave: u3 from param.f, velocity from wave vector
        k = param.k
        u3 = T(param.f(param.c, k, x, time))
        kmod = sqrt(T(k[1])^2 + T(k[2])^2)
        uR = SVector(-T(k[1]) * u3 / kmod, -T(k[2]) * u3 / kmod, u3)
    else            # far field
        uR = ui
    end
    return wavei_roe_pt(uL, uR, n, x, param, time)
end

@inline function wavev_pt(u::SVector{3, T}, x::SVector{2, T}, param, time) where {T}
    c = T(param.c)
    fx = SVector(-c * u[3], zero(T), -c * u[1])
    fy = SVector(zero(T), -c * u[3], -c * u[2])
    return fx, fy
end

"""
    mkapp_wave_pt(; c, k=nothing, f=nothing, bcm=nothing, bcs=nothing, src=nothing)

Wave-equation app in the pointwise (GPU-capable) convention. `c` is the wave
speed; `k` (wave vector) and `f(c, k, x, t)` are only needed for the
incoming-wave boundary type (ib = 3).
"""
mkapp_wave_pt(; c, k=nothing, f=nothing, bcm=nothing, bcs=nothing, src=nothing) =
    App(; nc=3, pg=false, arg=(; c, k, f), bcm, bcs, src,
        finvi=wavei_roe_pt, finvb=waveb_pt, finvv=wavev_pt)

# ----------------------------------------------------------------------- euler

@inline function euleri_roe_pt(uL::SVector{4, T}, uR::SVector{4, T}, n::SVector{2, T},
                               x::SVector{2, T}, param, time) where {T}
    gam = T(param.gamma)
    gam1 = gam - one(T)
    nx, ny = n

    rl, rul, rvl, rEl = uL
    rr, rur, rvr, rEr = uR

    rr1 = inv(rr)
    ur = rur * rr1
    vr = rvr * rr1
    Er = rEr * rr1
    u2r = ur^2 + vr^2
    pr = gam1 * (rEr - T(0.5) * rr * u2r)
    hr = Er + pr * rr1
    unr = ur * nx + vr * ny

    rl1 = inv(rl)
    ul = rul * rl1
    vl = rvl * rl1
    El = rEl * rl1
    u2l = ul^2 + vl^2
    pl = gam1 * (rEl - T(0.5) * rl * u2l)
    hl = El + pl * rl1
    unl = ul * nx + vl * ny

    # central flux
    f1 = T(0.5) * (rr * unr + rl * unl)
    f2 = T(0.5) * ((rur * unr + rul * unl) + nx * (pr + pl))
    f3 = T(0.5) * ((rvr * unr + rvl * unl) + ny * (pr + pl))
    f4 = T(0.5) * (rr * hr * unr + rl * hl * unl)

    # Roe averages
    di = sqrt(rr * rl1)
    d1 = inv(di + one(T))
    ua = (di * ur + ul) * d1
    va = (di * vr + vl) * d1
    ha = (di * hr + hl) * d1
    ci2 = gam1 * (ha - T(0.5) * (ua^2 + va^2))
    ci = sqrt(ci2)
    af = T(0.5) * (ua^2 + va^2)
    una = ua * nx + va * ny

    dr = rr - rl
    dru = rur - rul
    drv = rvr - rvl
    drE = rEr - rEl

    rlam1 = abs(una + ci)
    rlam2 = abs(una - ci)
    rlam3 = abs(una)
    s1 = T(0.5) * (rlam1 + rlam2)
    s2 = T(0.5) * (rlam1 - rlam2)
    al1x = gam1 * (af * dr - ua * dru - va * drv + drE)
    al2x = -una * dr + dru * nx + drv * ny
    cc1 = (s1 - rlam3) * al1x / ci2 + s2 * al2x / ci
    cc2 = s2 * al1x / ci + (s1 - rlam3) * al2x

    return SVector(f1 - T(0.5) * (rlam3 * dr + cc1),
                   f2 - T(0.5) * (rlam3 * dru + cc1 * ua + cc2 * nx),
                   f3 - T(0.5) * (rlam3 * drv + cc1 * va + cc2 * ny),
                   f4 - T(0.5) * (rlam3 * drE + cc1 * ha + cc2 * una))
end

@inline function eulerb_pt(uL::SVector{4, T}, n::SVector{2, T}, ib, ui::SVector{4, T},
                           x::SVector{2, T}, param, time) where {T}
    if ib == 2  # solid wall (reflection)
        un = uL[2] * n[1] + uL[3] * n[2]
        uR = SVector(uL[1], uL[2] - 2un * n[1], uL[3] - 2un * n[2], uL[4])
    else        # far field (ib == 1 and others)
        uR = ui
    end
    return euleri_roe_pt(uL, uR, n, x, param, time)
end

@inline function eulerv_pt(u::SVector{4, T}, x::SVector{2, T}, param, time) where {T}
    gam = T(param.gamma)
    ρ, ρu, ρv, E = u
    uu = ρu / ρ
    vv = ρv / ρ
    p = (gam - one(T)) * (E - T(0.5) * (ρu * uu + ρv * vv))
    fx = SVector(ρu, ρu * uu + p, ρv * uu, uu * (E + p))
    fy = SVector(ρv, ρu * vv, ρv * vv + p, vv * (E + p))
    return fx, fy
end

"""
    mkapp_euler_pt(; gamma=1.4, bcm=nothing, bcs=nothing, src=nothing)

Euler-equations app (Roe flux) in the pointwise (GPU-capable) convention.
"""
mkapp_euler_pt(; gamma=1.4, bcm=nothing, bcs=nothing, src=nothing) =
    App(; nc=4, pg=false, arg=(; gamma), bcm, bcs, src,
        finvi=euleri_roe_pt, finvb=eulerb_pt, finvv=eulerv_pt)

# ------------------------------------------------- convection-diffusion (LDG)

# Boundary types follow the legacy convection-diffusion app:
#   ib == 1  Dirichlet (u = 0 on the boundary)
#   ib == 2  Neumann

@inline cdinvi_pt(uL::SVector{1}, uR::SVector{1}, n, x, param, time) =
    convectioni_pt(uL, uR, n, x, param, time)

@inline function cdinvb_pt(uL::SVector{1}, n, ib, ui, x, param, time)
    uR = ib == 1 ? zero(uL) : uL
    return convectioni_pt(uL, uR, n, x, param, time)
end

@inline cdinvv_pt(u::SVector{1}, x, param, time) = convectionv_pt(u, x, param, time)

@inline function cdvisi_pt(uL::SVector{1, T}, uR::SVector{1, T},
                           qL::SMatrix{2, 1, T}, qR::SMatrix{2, 1, T},
                           n::SVector{2, T}, x::SVector{2, T}, param, time) where {T}
    κ = T(param.kappa)
    c11int = T(param.c11int)
    return SVector(-κ * (qR[1, 1] * n[1] + qR[2, 1] * n[2]) + c11int * (uL[1] - uR[1]))
end

@inline function cdvisb_pt(uL::SVector{1, T}, qL::SMatrix{2, 1, T}, n::SVector{2, T},
                           ib, ui::SVector{1, T}, x::SVector{2, T}, param, time) where {T}
    if ib == 1      # Dirichlet
        κ = T(param.kappa)
        c11 = T(param.c11)
        return SVector(-κ * (qL[1, 1] * n[1] + qL[2, 1] * n[2]) + c11 * (uL[1] - ui[1]))
    else            # Neumann
        return zero(uL)
    end
end

@inline function cdvisv_pt(u::SVector{1, T}, q::SMatrix{2, 1, T},
                           x::SVector{2, T}, param, time) where {T}
    κ = T(param.kappa)
    return SVector(-κ * q[1, 1]), SVector(-κ * q[2, 1])
end

@inline cdvisub_pt(uL::SVector{1}, n, ib, ui, x, param, time) =
    ib == 1 ? zero(uL) : uL

"""
    mkapp_convection_diffusion_pt(vf; kappa, c11, c11int=0, bcm=nothing, bcs=nothing, src=nothing)

Linear convection-diffusion app (LDG viscous terms) in the pointwise
(GPU-capable) convention. `vf` is a constant `SVector{2}` velocity or a
callable `x::SVector{2} -> SVector{2}`; `kappa` is the diffusivity and
`c11`/`c11int` the boundary/interior LDG stabilization coefficients.
"""
mkapp_convection_diffusion_pt(vf; kappa, c11, c11int=zero(kappa),
                              bcm=nothing, bcs=nothing, src=nothing) =
    App(; nc=1, pg=true, arg=(; vf, kappa, c11, c11int), bcm, bcs, src,
        finvi=cdinvi_pt, finvb=cdinvb_pt, finvv=cdinvv_pt,
        fvisi=cdvisi_pt, fvisb=cdvisb_pt, fvisv=cdvisv_pt, fvisub=cdvisub_pt)
