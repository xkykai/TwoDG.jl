"""
    ConvectionDiffusionEquation(velocity, κ)

Linear convection-diffusion

    ∂u/∂t + ∇ ⋅ (v u) = ∇ ⋅ (κ ∇u) + s,

with `velocity` a constant 2-vector or a callable `x -> SVector{2}` and
diffusivity `κ`. Discretized with LDG viscous fluxes in DG (stabilized by an
[`LDGStabilization`](@ref) policy passed to the problem) and with HDG for
steady problems.
"""
struct ConvectionDiffusionEquation{V, T} <: AbstractEquation
    velocity :: V
    κ        :: T
end
ConvectionDiffusionEquation(v::AbstractVector, κ) =
    ConvectionDiffusionEquation{SVector{2, float(eltype(v))}, typeof(κ)}(SVector{2}(v...), κ)

nvariables(::ConvectionDiffusionEquation) = 1
varnames(::ConvectionDiffusionEquation) = (:u,)
has_diffusion(::ConvectionDiffusionEquation) = true
default_numerical_flux(::ConvectionDiffusionEquation) = LaxFriedrichs()

@inline function flux(eq::ConvectionDiffusionEquation, u::SVector{1, T}, x, t) where {T}
    v = velocity_at(eq.velocity, x)
    return convert(T, v[1]) * u, convert(T, v[2]) * u
end

@inline function max_abs_speed(eq::ConvectionDiffusionEquation, u::SVector{1, T},
                               n, x, t) where {T}
    v = velocity_at(eq.velocity, x)
    return abs(convert(T, v[1]) * n[1] + convert(T, v[2]) * n[2])
end

"""
    PoissonEquation(κ=1.0)

Poisson / pure-diffusion equation `-∇ ⋅ (κ ∇u) = s`, for `HDGProblem` and
`CGProblem`.
"""
struct PoissonEquation{T} <: AbstractEquation
    κ :: T
end
PoissonEquation() = PoissonEquation(1.0)

nvariables(::PoissonEquation) = 1
varnames(::PoissonEquation) = (:u,)
has_diffusion(::PoissonEquation) = true

# ------------------------------------------------------- LDG viscous terms

"""
    LDGStabilization(c11=1.0, c11int=0.0)

Penalty coefficients of the LDG viscous fluxes (Cockburn & Shu, SINUM 35,
1998): `c11` on Dirichlet boundary faces, `c11int` on interior faces. A
swappable policy object — pass to `DGProblem(...; stabilization=...)`.
"""
struct LDGStabilization{T}
    c11    :: T
    c11int :: T
end
LDGStabilization(c11=1.0, c11int=0.0) = LDGStabilization(promote(c11, c11int)...)

default_stabilization(::ConvectionDiffusionEquation) = LDGStabilization()
default_stabilization(::PoissonEquation) = LDGStabilization()

@inline function viscous_flux(eq::Union{ConvectionDiffusionEquation, PoissonEquation},
                              u::SVector{1, T}, q, x, t) where {T}
    κ = convert(T, eq.κ)
    return SVector(-κ * q[1, 1]), SVector(-κ * q[2, 1])
end

"""
    viscous_numerical_flux(stab, eq, uL, uR, qL, qR, n, x, t) -> SVector

LDG viscous interface flux: the alternating choice takes the trace û from the
left and the gradient from the *right* element, plus the interior penalty:
`F̂ᵥ = -κ qR ⋅ n + c11int (uL - uR)`.
"""
@inline function viscous_numerical_flux(stab::LDGStabilization,
                                        eq::Union{ConvectionDiffusionEquation, PoissonEquation},
                                        uL::SVector{1, T}, uR, qL, qR, n, x, t) where {T}
    κ = convert(T, eq.κ)
    c11int = convert(T, stab.c11int)
    return SVector(-κ * (qR[1, 1] * n[1] + qR[2, 1] * n[2]) + c11int * (uL[1] - uR[1]))
end

"""
    boundary_viscous_flux(bc, stab, eq, uL, qL, n, x, t) -> SVector

Viscous flux on a boundary face: for [`Dirichlet`](@ref) data `g`,
`-κ qL ⋅ n + c11 (uL - g)`; for [`Neumann`](@ref), the prescribed flux
(currently homogeneous).
"""
@inline function boundary_viscous_flux(bc::Dirichlet, stab::LDGStabilization,
                                       eq, uL::SVector{1, T}, qL, n, x, t) where {T}
    κ = convert(T, eq.κ)
    c11 = convert(T, stab.c11)
    g = bc_data(bc.value, uL, x, t)
    return SVector(-κ * (qL[1, 1] * n[1] + qL[2, 1] * n[2]) + c11 * (uL[1] - g[1]))
end

@inline boundary_viscous_flux(bc::Neumann, stab, eq, uL::SVector{1}, qL, n, x, t) =
    zero(uL)
