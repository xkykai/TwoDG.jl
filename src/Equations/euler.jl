"""
    EulerEquations(; γ=1.4)

Compressible Euler equations for the conserved state `u = (ρ, ρu, ρv, ρE)`,
with ideal-gas ratio of specific heats `γ`. Default numerical flux:
[`RoeFlux`](@ref).
"""
struct EulerEquations{T} <: AbstractEquation
    γ :: T
end
EulerEquations(; γ=1.4) = EulerEquations(γ)

nvariables(::EulerEquations) = 4
varnames(::EulerEquations) = (:ρ, :ρu, :ρv, :ρE)
default_numerical_flux(::EulerEquations) = RoeFlux()

# ------------------------------------------------------------- primitives
# Defined once; the physical flux, the Roe flux, boundary states, and the
# derived-quantity fields below all call these.

"Density `ρ` of a conserved Euler state."
@inline density(eq::EulerEquations, u::SVector{4}) = u[1]

"Velocity vector `(u, v) = (ρu, ρv)/ρ` of a conserved Euler state."
@inline velocity(eq::EulerEquations, u::SVector{4}) = SVector(u[2], u[3]) / u[1]

"Ideal-gas pressure `p = (γ-1)(ρE - ρ|v|²/2)` of a conserved Euler state."
@inline function pressure(eq::EulerEquations, u::SVector{4, T}) where {T}
    γ = convert(T, eq.γ)
    return (γ - one(T)) * (u[4] - (u[2]^2 + u[3]^2) / (2 * u[1]))
end

"Speed of sound `c = √(γp/ρ)`."
@inline soundspeed(eq::EulerEquations, u::SVector{4, T}) where {T} =
    sqrt(convert(T, eq.γ) * pressure(eq, u) / u[1])

"Specific total enthalpy `h = (ρE + p)/ρ`."
@inline enthalpy(eq::EulerEquations, u::SVector{4}) = (u[4] + pressure(eq, u)) / u[1]

"Mach number `|v|/c`."
@inline mach(eq::EulerEquations, u::SVector{4}) =
    norm(velocity(eq, u)) / soundspeed(eq, u)

"Entropy function `s = p/ρ^γ`."
@inline entropy(eq::EulerEquations, u::SVector{4, T}) where {T} =
    pressure(eq, u) / u[1]^convert(T, eq.γ)

# ------------------------------------------------------------------ fluxes

@inline function flux(eq::EulerEquations, u::SVector{4, T}, x, t) where {T}
    ρ, ρu, ρv, ρE = u
    v = velocity(eq, u)
    p = pressure(eq, u)
    fx = SVector(ρu, ρu * v[1] + p, ρv * v[1], v[1] * (ρE + p))
    fy = SVector(ρv, ρu * v[2], ρv * v[2] + p, v[2] * (ρE + p))
    return fx, fy
end

@inline function max_abs_speed(eq::EulerEquations, u::SVector{4}, n, x, t)
    v = velocity(eq, u)
    return abs(v[1] * n[1] + v[2] * n[2]) + soundspeed(eq, u)
end

# Roe (1981), JCP 43:357. F̂ = ½(F(uL)+F(uR))·n − ½|Â(ũ)|(uR−uL); the
# dissipation is |Â| applied to the jump, evaluated at the Roe average.
@inline function (::RoeFlux)(eq::EulerEquations, uL::SVector{4, T}, uR::SVector{4, T},
                             n, x, t) where {T}
    central = (normal_flux(eq, uL, n, x, t) + normal_flux(eq, uR, n, x, t)) / 2
    return central - roe_dissipation(eq, uL, uR, n) / 2
end

# Density-weighted (Roe) average of velocity and total enthalpy: the unique
# state ũ with Â(ũ)(uR − uL) = F(uR) − F(uL).
@inline function roe_average(eq::EulerEquations, uL::SVector{4, T},
                             uR::SVector{4, T}) where {T}
    d = sqrt(uR[1] / uL[1])
    w = inv(d + one(T))
    vL = velocity(eq, uL)
    vR = velocity(eq, uR)
    ũ = (d * vR[1] + vL[1]) * w
    ṽ = (d * vR[2] + vL[2]) * w
    h̃ = (d * enthalpy(eq, uR) + enthalpy(eq, uL)) * w
    return ũ, ṽ, h̃
end

# |Â(ũ)| (uR − uL): eigenvalues |ũ·n ± c̃| (acoustic) and |ũ·n| (entropy/shear),
# with the acoustic/normal-momentum wave strengths α below.
@inline function roe_dissipation(eq::EulerEquations, uL::SVector{4, T},
                                 uR::SVector{4, T}, n) where {T}
    γ1 = convert(T, eq.γ) - one(T)
    ũ, ṽ, h̃ = roe_average(eq, uL, uR)
    ke = (ũ^2 + ṽ^2) / 2
    c̃² = γ1 * (h̃ - ke)
    c̃ = sqrt(c̃²)
    un = ũ * n[1] + ṽ * n[2]
    Δ = uR - uL

    λ⁺ = abs(un + c̃)                 # right-going acoustic
    λ⁻ = abs(un - c̃)                 # left-going acoustic
    λ⁰ = abs(un)                     # entropy + shear
    s½ = (λ⁺ + λ⁻) / 2
    d½ = (λ⁺ - λ⁻) / 2

    # wave strengths: energy-like and normal-momentum projections of the jump
    α_E = γ1 * (ke * Δ[1] - ũ * Δ[2] - ṽ * Δ[3] + Δ[4])
    α_n = -un * Δ[1] + Δ[2] * n[1] + Δ[3] * n[2]
    c₁ = (s½ - λ⁰) * α_E / c̃² + d½ * α_n / c̃
    c₂ = d½ * α_E / c̃ + (s½ - λ⁰) * α_n

    return SVector(λ⁰ * Δ[1] + c₁,
                   λ⁰ * Δ[2] + c₁ * ũ + c₂ * n[1],
                   λ⁰ * Δ[3] + c₁ * ṽ + c₂ * n[2],
                   λ⁰ * Δ[4] + c₁ * h̃ + c₂ * un)
end

# slip wall: reflect the normal momentum
@inline function boundary_state(::SlipWall, eq::EulerEquations, uL::SVector{4, T},
                                n, x, t) where {T}
    ρvn = uL[2] * n[1] + uL[3] * n[2]
    return SVector(uL[1], uL[2] - 2ρvn * n[1], uL[3] - 2ρvn * n[2], uL[4])
end

# --------------------------------------------------------- derived fields

"""
    derived_field(f, eq, u) -> Matrix

Evaluate a pointwise derived quantity `f(eq, u::SVector) -> Real` (e.g.
[`pressure`](@ref), [`mach`](@ref)) over a solution field `u (npl, nc, nt)`;
returns `(npl, nt)`.
"""
derived_field(f::F, eq::AbstractEquation, u::AbstractArray{<:Any, 3}) where {F} =
    _derived_field(f, eq, u, Val(nvariables(eq)))

function _derived_field(f::F, eq, u::AbstractArray{T, 3}, ::Val{NC}) where {F, T, NC}
    npl, _, nt = size(u)
    sca = Matrix{T}(undef, npl, nt)
    @inbounds for j in 1:nt, i in 1:npl
        sca[i, j] = f(eq, SVector{NC, T}(ntuple(c -> u[i, c, j], Val(NC))))
    end
    return sca
end

"""
    eulereval(u, str, γ) -> Matrix

String-keyed Euler derived quantities for plotting scripts — a thin lookup
over the dispatched primitives ([`density`](@ref), [`pressure`](@ref),
[`mach`](@ref), …): `"r"`, `"u"`, `"v"`, `"p"`, `"c"`, `"M"`, `"s"`, and the
1D characteristic combinations `"Jp"`/`"Jm"` (which, as in the original
plotting scripts, read component 2 as the Riemann `u` rather than dividing
by density).
"""
function eulereval(u::AbstractArray{T, 3}, str, γ) where {T}
    eq = EulerEquations(convert(T, γ))
    str == "r" && return u[:, 1, :]
    f = if str == "u"
        (e, v) -> velocity(e, v)[1]
    elseif str == "v"
        (e, v) -> velocity(e, v)[2]
    elseif str == "p"
        pressure
    elseif str == "c"
        soundspeed
    elseif str == "M"
        mach
    elseif str == "s"
        entropy
    elseif str == "Jp"
        (e, v) -> v[2] + 2 * soundspeed(e, v) / (convert(T, e.γ) - one(T))
    elseif str == "Jm"
        (e, v) -> v[2] - 2 * soundspeed(e, v) / (convert(T, e.γ) - one(T))
    else
        error("Unknown quantity: $str")
    end
    return derived_field(f, eq, u)
end

"""
    riemann_to_canonical(v, s, J⁺, J⁻, γ) -> (ρ, ρu₁, ρu₂, ρE)

Convert the 1D Riemann-invariant variables — tangential velocity `v`, entropy
`s = p/ρ^γ`, and invariants `J± = u ± 2c/(γ-1)` — to conserved Euler
variables. Inverse of [`canonical_to_riemann`](@ref); used to prescribe
characteristic far-field states.
"""
function riemann_to_canonical(v, s, J⁺, J⁻, γ)
    c = (γ - 1) / 4 * (J⁺ - J⁻)
    ρu₁ = (J⁺ + J⁻) / 2
    ρ = (c^2 / γ / s)^(1 / (γ - 1))
    ρu₂ = ρ * v
    p = s * ρ^γ
    ρE = p / (γ - 1) + 0.5 * (ρu₁^2 + ρu₂^2) / ρ
    return ρ, ρu₁, ρu₂, ρE
end

"""
    canonical_to_riemann(ρ, ρu₁, ρu₂, ρE, γ) -> (v, s, J⁺, J⁻)

Convert conserved Euler variables to the 1D Riemann-invariant variables:
tangential velocity `v`, entropy `s = p/ρ^γ`, and invariants
`J± = u ± 2c/(γ-1)`. See [`riemann_to_canonical`](@ref).
"""
function canonical_to_riemann(ρ, ρu₁, ρu₂, ρE, γ)
    p = (γ - 1) * (ρE - 0.5 * (ρu₁^2 + ρu₂^2) / ρ)
    v = ρu₂ / ρ
    s = p / (ρ^γ)
    c = sqrt(γ * p / ρ)
    J⁺ = ρu₁ + 2c / (γ - 1)
    J⁻ = ρu₁ - 2c / (γ - 1)
    return v, s, J⁺, J⁻
end
