"""
    ConvectionEquation(velocity)

Linear scalar convection

    ∂u/∂t + ∇ ⋅ (v u) = s,

with `velocity` either a constant 2-vector or a callable
`x::SVector{2} -> SVector{2}` for a spatially varying field.
"""
struct ConvectionEquation{V} <: AbstractEquation
    velocity :: V
end
ConvectionEquation(v::AbstractVector) =
    ConvectionEquation{SVector{2, float(eltype(v))}}(SVector{2}(v...))

nvariables(::ConvectionEquation) = 1
varnames(::ConvectionEquation) = (:u,)
default_numerical_flux(::ConvectionEquation) = LaxFriedrichs()

# velocity field: constant SVector or callable x -> SVector
@inline velocity_at(v::SVector{2}, x) = v
@inline velocity_at(v, x) = v(x)

@inline function flux(eq::ConvectionEquation, u::SVector{1, T}, x, t) where {T}
    v = velocity_at(eq.velocity, x)
    return convert(T, v[1]) * u, convert(T, v[2]) * u
end

@inline function max_abs_speed(eq::ConvectionEquation, u::SVector{1, T}, n, x, t) where {T}
    v = velocity_at(eq.velocity, x)
    return abs(convert(T, v[1]) * n[1] + convert(T, v[2]) * n[2])
end
