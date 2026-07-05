"""
    WaveEquation(c; k=nothing, f=nothing)

First-order acoustic wave system for `u = (q₁, …, q_Dim, p)`,

    ∂q/∂t + c ∇p = 0,    ∂p/∂t + c ∇ ⋅ q = 0,

with speed `c` (`Dim + 1` components; 2D by default, or inferred from the
wave vector `k`). `k` and `f(c, k, x, t)` prescribe the incident field and
are only needed when an [`IncomingWave`](@ref) boundary is used.
"""
struct WaveEquation{Dim, T, K, F} <: AbstractEquation{Dim}
    c :: T
    k :: K
    f :: F
end
function WaveEquation(c; k=nothing, f=nothing)
    kv = k === nothing ? nothing : SVector{length(k)}(k...)
    Dim = k === nothing ? 2 : length(k)
    return WaveEquation{Dim, typeof(c), typeof(kv), typeof(f)}(c, kv, f)
end

nvariables(::WaveEquation{Dim}) where {Dim} = Dim + 1
varnames(::WaveEquation{2}) = (:q1, :q2, :p)
varnames(::WaveEquation{3}) = (:q1, :q2, :q3, :p)
default_numerical_flux(::WaveEquation) = RoeFlux()

@inline function flux(eq::WaveEquation{2}, u::SVector{3, T}, x, t) where {T}
    c = convert(T, eq.c)
    fx = SVector(-c * u[3], zero(T), -c * u[1])
    fy = SVector(zero(T), -c * u[3], -c * u[2])
    return fx, fy
end

@inline max_abs_speed(eq::WaveEquation, u::SVector{3, T}, n, x, t) where {T} =
    abs(convert(T, eq.c))

# Exact upwind flux of the linear system: the characteristic variables along n
# are p ± q·n with speeds ±c (and a stationary transverse mode), so
# |Â|(uR − uL) works out to |c| times the projections below.
@inline function (::RoeFlux)(eq::WaveEquation, uL::SVector{3, T}, uR::SVector{3, T},
                             n, x, t) where {T}
    central = (normal_flux(eq, uL, n, x, t) + normal_flux(eq, uR, n, x, t)) / 2
    ca = abs(convert(T, eq.c))
    Δqn = (uL[1] - uR[1]) * n[1] + (uL[2] - uR[2]) * n[2]   # jump of q ⋅ n
    Δp = uL[3] - uR[3]                                       # jump of p
    dissipation = ca / 2 * SVector(Δqn * n[1], Δqn * n[2], Δp)
    return central + dissipation
end

# solid wall: reflect the normal component of q, keep p
@inline function boundary_state(::SlipWall, eq::WaveEquation, uL::SVector{3, T},
                                n, x, t) where {T}
    qn = uL[1] * n[1] + uL[2] * n[2]
    return SVector(uL[1] - 2qn * n[1], uL[2] - 2qn * n[2], uL[3])
end

# incoming wave: p from the prescribed incident field, q along the wave vector
@inline function boundary_state(::IncomingWave, eq::WaveEquation, uL::SVector{3, T},
                                n, x, t) where {T}
    k = eq.k
    p = convert(T, eq.f(eq.c, k, x, t))
    kmod = sqrt(convert(T, k[1])^2 + convert(T, k[2])^2)
    return SVector(-convert(T, k[1]) * p / kmod, -convert(T, k[2]) * p / kmod, p)
end
