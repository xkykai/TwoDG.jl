"""
    ConvectionDiffusionEquation(velocity, κ)
    ConvectionDiffusionEquation{Dim}(velocity, κ)

Linear convection-diffusion

    ∂u/∂t + ∇ ⋅ (v u) = ∇ ⋅ (κ ∇u) + s,

with `velocity` a constant `Dim`-vector (dimension inferred from its length)
or a callable `x -> SVector{Dim}` (2D by default; use the explicit `{Dim}`
form in 3D), and diffusivity `κ`. Discretized with LDG viscous fluxes in DG
(stabilized by an [`LDGStabilization`](@ref) policy passed to the problem)
and with HDG for steady problems.
"""
struct ConvectionDiffusionEquation{Dim, V, T} <: AbstractEquation{Dim}
    velocity :: V
    κ        :: T
end
ConvectionDiffusionEquation(v::AbstractVector, κ) =
    ConvectionDiffusionEquation{length(v), SVector{length(v), float(eltype(v))}, typeof(κ)}(
        SVector{length(v)}(v...), κ)
ConvectionDiffusionEquation(v, κ) =
    ConvectionDiffusionEquation{2, typeof(v), typeof(κ)}(v, κ)
ConvectionDiffusionEquation{Dim}(v, κ) where {Dim} =
    ConvectionDiffusionEquation{Dim, typeof(v), typeof(κ)}(v, κ)

nvariables(::ConvectionDiffusionEquation) = 1
varnames(::ConvectionDiffusionEquation) = (:u,)
has_diffusion(::ConvectionDiffusionEquation) = true
default_numerical_flux(::ConvectionDiffusionEquation) = LaxFriedrichs()

@inline function flux(eq::ConvectionDiffusionEquation{Dim}, u::SVector{1, T},
                      x, t) where {Dim, T}
    v = velocity_at(eq.velocity, x)
    return ntuple(d -> convert(T, v[d]) * u, Val(Dim))
end

@inline function max_abs_speed(eq::ConvectionDiffusionEquation, u::SVector{1, T},
                               n, x, t) where {T}
    v = velocity_at(eq.velocity, x)
    return abs(normal_velocity(v, n, T))
end

"""
    PoissonEquation(κ=1.0)
    PoissonEquation{Dim}(κ=1.0)

Poisson / pure-diffusion equation `-∇ ⋅ (κ ∇u) = s`, for `HDGProblem` and
`CGProblem`. Dimension-polymorphic: 2D by default (there is no directional
parameter to infer from), `PoissonEquation{3}()` in 3D.
"""
struct PoissonEquation{Dim, T} <: AbstractEquation{Dim}
    κ :: T
end
PoissonEquation(κ=1.0) = PoissonEquation{2, typeof(κ)}(κ)
PoissonEquation{Dim}(κ=1.0) where {Dim} = PoissonEquation{Dim, typeof(κ)}(κ)

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

# q ⋅ n of the first (scalar) component's gradient, unrolled over Dim
@inline function _grad_dot_n(q, n::SVector{Dim}) where {Dim}
    s = q[1, 1] * n[1]
    for d in 2:Dim
        s += q[d, 1] * n[d]
    end
    return s
end

@inline function viscous_flux(eq::Union{ConvectionDiffusionEquation{Dim}, PoissonEquation{Dim}},
                              u::SVector{1, T}, q, x, t) where {Dim, T}
    κ = convert(T, eq.κ)
    return ntuple(d -> SVector(-κ * q[d, 1]), Val(Dim))
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
    return SVector(-κ * _grad_dot_n(qR, n) + c11int * (uL[1] - uR[1]))
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
    return SVector(-κ * _grad_dot_n(qL, n) + c11 * (uL[1] - g[1]))
end

@inline boundary_viscous_flux(bc::Neumann, stab, eq, uL::SVector{1}, qL, n, x, t) =
    zero(uL)
