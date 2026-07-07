"""
    EulerEquations(; γ=1.4)
    EulerEquations{Dim}(; γ=1.4)

Compressible Euler equations for the conserved state `(ρ, ρu, ρv[, ρw], ρE)`
(`Dim + 2` components; 2D by default), with ideal-gas ratio of specific heats
`γ`. Default numerical flux: [`RoeFlux`](@ref).
"""
struct EulerEquations{Dim, T} <: AbstractEquation{Dim}
    γ :: T
end
EulerEquations(γ::Real) = EulerEquations{2, typeof(γ)}(γ)
EulerEquations(; γ=1.4) = EulerEquations(γ)
EulerEquations{Dim}(γ::Real) where {Dim} = EulerEquations{Dim, typeof(γ)}(γ)
EulerEquations{Dim}(; γ=1.4) where {Dim} = EulerEquations{Dim}(γ)

nvariables(::EulerEquations{Dim}) where {Dim} = Dim + 2
varnames(::EulerEquations{2}) = (:ρ, :ρu, :ρv, :ρE)
varnames(::EulerEquations{3}) = (:ρ, :ρu, :ρv, :ρw, :ρE)
default_numerical_flux(::EulerEquations) = RoeFlux()

# ------------------------------------------------------------- primitives
# Defined once, for any `Dim`, on the conserved state `(ρ, ρv..., ρE)` with
# `NC = Dim + 2` components; the physical flux, the Roe flux, boundary states,
# and the derived-quantity fields below all call these. The accumulation
# helpers keep the momentum sums left-associated (in component order) so the
# 2D results are bit-identical to the pre-3D specialized methods.

# In-order accumulation helpers. These exist for two reasons: the sums must
# associate left-to-right in component order (so the Dim = 2 results are
# bit-identical to the pre-3D code), and locals that a closure captures must
# be assigned exactly once — a captured local reassigned in a loop gets boxed,
# which is a silent 100× cliff inside a KA kernel. Callers bind each helper's
# result once and only capture that.
@inline function _minus_mom_dot(s, v::SVector{Dim}, Δ) where {Dim}
    for d in 1:Dim
        s -= v[d] * Δ[d + 1]
    end
    return s
end
@inline function _plus_mom_dot_n(s, Δ, n::SVector{Dim}) where {Dim}
    for d in 1:Dim
        s += Δ[d + 1] * n[d]
    end
    return s
end
@inline function _dotn(v::SVector{Dim}, n) where {Dim}
    s = v[1] * n[1]
    for d in 2:Dim
        s += v[d] * n[d]
    end
    return s
end

"Density `ρ` of a conserved Euler state."
@inline density(eq::EulerEquations, u::SVector) = u[1]

"Velocity vector `ρv/ρ` of a conserved Euler state."
@inline velocity(eq::EulerEquations{Dim}, u::SVector) where {Dim} =
    SVector(ntuple(d -> u[d + 1], Val(Dim))) / u[1]

"Ideal-gas pressure `p = (γ-1)(ρE - ρ|v|²/2)` of a conserved Euler state."
@inline function pressure(eq::EulerEquations{Dim}, u::SVector{NC, T}) where {Dim, NC, T}
    γ = convert(T, eq.γ)
    ρv² = u[2]^2
    for d in 2:Dim
        ρv² += u[d + 1]^2
    end
    return (γ - one(T)) * (u[NC] - ρv² / (2 * u[1]))
end

"Speed of sound `c = √(γp/ρ)`."
@inline soundspeed(eq::EulerEquations, u::SVector{NC, T}) where {NC, T} =
    sqrt(convert(T, eq.γ) * pressure(eq, u) / u[1])

"Specific total enthalpy `h = (ρE + p)/ρ`."
@inline enthalpy(eq::EulerEquations, u::SVector{NC}) where {NC} =
    (u[NC] + pressure(eq, u)) / u[1]

"Mach number `|v|/c`."
@inline mach(eq::EulerEquations, u::SVector) =
    norm(velocity(eq, u)) / soundspeed(eq, u)

"Entropy function `s = p/ρ^γ`."
@inline entropy(eq::EulerEquations, u::SVector{NC, T}) where {NC, T} =
    pressure(eq, u) / u[1]^convert(T, eq.γ)

"Kinetic energy density `ρ|v|²/2` of a conserved Euler state."
@inline function energy_kinetic(eq::EulerEquations{Dim}, u::SVector) where {Dim}
    ρv² = u[2]^2
    for d in 2:Dim
        ρv² += u[d + 1]^2
    end
    return ρv² / (2 * u[1])
end

"Total energy density `ρE` of a conserved Euler state."
@inline energy_total(eq::EulerEquations, u::SVector{NC}) where {NC} = u[NC]

"Internal energy density `ρe = ρE - ρ|v|²/2` of a conserved Euler state."
@inline energy_internal(eq::EulerEquations, u::SVector{NC}) where {NC} =
    u[NC] - energy_kinetic(eq, u)

@inline wavespeed(eq::EulerEquations, u::SVector) =
    norm(velocity(eq, u)) + soundspeed(eq, u)

# ------------------------------------------------------------------ fluxes

# f_d = (ρ v_d, ρv v_d + p e_d, v_d (ρE + p)) — one definition for every Dim
@inline function flux(eq::EulerEquations{Dim}, u::SVector{NC, T}, x, t) where {Dim, NC, T}
    v = velocity(eq, u)
    p = pressure(eq, u)
    ρE = u[NC]
    return ntuple(Val(Dim)) do d
        SVector(ntuple(Val(NC)) do c
            c == 1  ? u[d + 1] :
            c == NC ? v[d] * (ρE + p) :
            c == d + 1 ? u[c] * v[d] + p : u[c] * v[d]
        end)
    end
end

@inline function max_abs_speed(eq::EulerEquations{Dim}, u::SVector, n, x, t) where {Dim}
    v = velocity(eq, u)
    return abs(_dotn(v, n)) + soundspeed(eq, u)
end

# Roe (1981), JCP 43:357. F̂ = ½(F(uL)+F(uR))·n − ½|Â(ũ)|(uR−uL); the
# dissipation is |Â| applied to the jump, evaluated at the Roe average.
@inline function (::RoeFlux)(eq::EulerEquations, uL::SVector{NC, T}, uR::SVector{NC, T},
                             n, x, t) where {NC, T}
    central = (normal_flux(eq, uL, n, x, t) + normal_flux(eq, uR, n, x, t)) / 2
    return central - roe_dissipation(eq, uL, uR, n) / 2
end

# Density-weighted (Roe) average of velocity and total enthalpy: the unique
# state ũ with Â(ũ)(uR − uL) = F(uR) − F(uL).
@inline function roe_average(eq::EulerEquations{Dim}, uL::SVector{NC, T},
                             uR::SVector{NC, T}) where {Dim, NC, T}
    d = sqrt(uR[1] / uL[1])
    w = inv(d + one(T))
    vL = velocity(eq, uL)
    vR = velocity(eq, uR)
    ṽ = SVector(ntuple(k -> (d * vR[k] + vL[k]) * w, Val(Dim)))
    h̃ = (d * enthalpy(eq, uR) + enthalpy(eq, uL)) * w
    return ṽ, h̃
end

# |Â(ũ)| (uR − uL): eigenvalues |ũ·n ± c̃| (acoustic) and |ũ·n| (entropy +
# Dim−1 shear), with the acoustic/normal-momentum wave strengths α below.
@inline function roe_dissipation(eq::EulerEquations{Dim}, uL::SVector{NC, T},
                                 uR::SVector{NC, T}, n) where {Dim, NC, T}
    γ1 = convert(T, eq.γ) - one(T)
    ṽ, h̃ = roe_average(eq, uL, uR)
    ke = _dotn(ṽ, ṽ) / 2
    c̃² = γ1 * (h̃ - ke)
    c̃ = sqrt(c̃²)
    un = _dotn(ṽ, n)
    Δ = uR - uL

    λ⁺ = abs(un + c̃)                 # right-going acoustic
    λ⁻ = abs(un - c̃)                 # left-going acoustic
    λ⁰ = abs(un)                     # entropy + shear
    s½ = (λ⁺ + λ⁻) / 2
    d½ = (λ⁺ - λ⁻) / 2

    # wave strengths: energy-like and normal-momentum projections of the jump
    α_E = γ1 * (_minus_mom_dot(ke * Δ[1], ṽ, Δ) + Δ[NC])
    α_n = _plus_mom_dot_n(-un * Δ[1], Δ, n)
    c₁ = (s½ - λ⁰) * α_E / c̃² + d½ * α_n / c̃
    c₂ = d½ * α_E / c̃ + (s½ - λ⁰) * α_n

    return SVector(ntuple(Val(NC)) do c
        c == 1  ? λ⁰ * Δ[1] + c₁ :
        c == NC ? λ⁰ * Δ[NC] + c₁ * h̃ + c₂ * un :
                  λ⁰ * Δ[c] + c₁ * ṽ[c - 1] + c₂ * n[c - 1]
    end)
end

# slip wall: reflect the normal momentum
@inline function boundary_state(::SlipWall, eq::EulerEquations{Dim},
                                uL::SVector{NC, T}, n, x, t) where {Dim, NC, T}
    ρvn = _plus_mom_dot_n(zero(T), uL, n)
    return SVector(ntuple(Val(NC)) do c
        (c == 1 || c == NC) ? uL[c] : uL[c] - 2ρvn * n[c - 1]
    end)
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
