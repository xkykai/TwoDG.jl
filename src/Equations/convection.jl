"""
    ConvectionEquation(velocity)
    ConvectionEquation{Dim}(velocity)

Linear scalar convection

    ∂u/∂t + ∇ ⋅ (v u) = s,

with `velocity` either a constant `Dim`-vector (the dimension is inferred
from its length) or a callable `x::SVector{Dim} -> SVector{Dim}` for a
spatially varying field (2D by default; use `ConvectionEquation{3}(f)` in 3D).
"""
struct ConvectionEquation{Dim, V} <: AbstractEquation{Dim}
    velocity :: V
end
ConvectionEquation(v::AbstractVector) =
    ConvectionEquation{length(v), SVector{length(v), float(eltype(v))}}(SVector{length(v)}(v...))
ConvectionEquation(v) = ConvectionEquation{2, typeof(v)}(v)
ConvectionEquation{Dim}(v) where {Dim} = ConvectionEquation{Dim, typeof(v)}(v)

nvariables(::ConvectionEquation) = 1
varnames(::ConvectionEquation) = (:u,)
default_numerical_flux(::ConvectionEquation) = LaxFriedrichs()

# velocity field: constant SVector or callable x -> SVector
@inline velocity_at(v::SVector, x) = v
@inline velocity_at(v, x) = v(x)

# v ⋅ n in the working precision T (kernel-side states set T; the velocity
# field may be stored at a different precision)
@inline function normal_velocity(v, n::SVector{Dim}, ::Type{T}) where {Dim, T}
    vn = convert(T, v[1]) * n[1]
    for d in 2:Dim
        vn += convert(T, v[d]) * n[d]
    end
    return vn
end

@inline function flux(eq::ConvectionEquation{Dim}, u::SVector{1, T}, x, t) where {Dim, T}
    v = velocity_at(eq.velocity, x)
    return ntuple(d -> convert(T, v[d]) * u, Val(Dim))
end

@inline function max_abs_speed(eq::ConvectionEquation, u::SVector{1, T}, n, x, t) where {T}
    v = velocity_at(eq.velocity, x)
    return abs(normal_velocity(v, n, T))
end
