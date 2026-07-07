# Quadrature-exact functionals over a (running, possibly device-resident) DG
# field: one KernelAbstractions kernel writes a per-element partial sum, a
# device reduction adds them — only scalars cross back to the host. All
# kernels consume the compact VolumeTables layout through quad_weight /
# quad_coords; the dense ctx.wjac / ctx.pg properties materialize whole
# arrays and must not appear here.

# Non-capturing state constructors: an `ntuple` closure over a local that is
# later reassigned in the enclosing scope gets boxed, a silent 100× cliff
# inside a kernel. These helpers keep every captured local assigned-once.
@inline function _interp_state(u, shap, g, e, ::Val{NC}) where {NC}
    T = eltype(u)
    return SVector{NC, T}(ntuple(Val(NC)) do c
        s = zero(T)
        @inbounds for i in 1:size(shap, 1)
            s += shap[i, g] * u[i, c, e]
        end
        s
    end)
end

@inline _node_state(u, i, e, ::Val{NC}) where {NC} =
    SVector{NC, eltype(u)}(ntuple(c -> @inbounds(u[i, c, e]), Val(NC)))

@kernel function _integral_kernel!(out, f::F, eq, u, shap,
                                   vol::VolumeTables, ::Val{NC}) where {F, NC}
    e = @index(Global)
    T = eltype(out)
    acc = zero(T)
    for g in 1:size(shap, 2)
        ug = _interp_state(u, shap, g, e, Val(NC))
        acc += quad_weight(vol, g, e) * convert(T, f(eq, ug))
    end
    @inbounds out[e] = acc
end

@kernel function _component_integral_kernel!(out, u, shap, vol::VolumeTables,
                                             squared::Bool)
    e, c = @index(Global, NTuple)
    T = eltype(out)
    acc = zero(T)
    for g in 1:size(shap, 2)
        s = zero(T)
        @inbounds for i in 1:size(shap, 1)
            s += shap[i, g] * u[i, c, e]
        end
        acc += quad_weight(vol, g, e) * ifelse(squared, s * s, s)
    end
    @inbounds out[e, c] = acc
end

@kernel function _l2error_kernel!(out, exact::F, u, component::Int, t, shap,
                                  vol::VolumeTables, ::Val{Dim}) where {F, Dim}
    e = @index(Global)
    T = eltype(out)
    acc = zero(T)
    for g in 1:size(shap, 2)
        s = zero(T)
        @inbounds for i in 1:size(shap, 1)
            s += shap[i, g] * u[i, component, e]
        end
        x = quad_coords(vol, g, e, Val(Dim))
        d = s - convert(T, exact(x, t))
        acc += quad_weight(vol, g, e) * d * d
    end
    @inbounds out[e] = acc
end

@kernel function _wavespeed_kernel!(out, eq, u, ::Val{NC}) where {NC}
    e = @index(Global)
    T = eltype(out)
    m = zero(T)
    for i in 1:size(u, 1)
        m = max(m, convert(T, wavespeed(eq, _node_state(u, i, e, Val(NC)))))
    end
    @inbounds out[e] = m
end

"""
    integrate(f, eq, u, ctx) -> Real

Integral `∫_Ω f(eq, u(x)) dx` of a pointwise functional over a DG field
`u (npl, nc, nt)`, using the quadrature of the geometry cache
`ctx::GeometricFactors`. `f` is any callable `(eq, u::SVector) -> Real` —
the same contract as [`derived_field`](@ref) — e.g. [`pressure`](@ref),
[`energy_kinetic`](@ref), [`entropy`](@ref), or a user closure. Runs as a
KernelAbstractions kernel plus a device reduction, so `u` may live on the
GPU; only the scalar result crosses to the host.
"""
function integrate(f::F, eq::AbstractEquation, u::AbstractArray{T, 3},
                   ctx::GeometricFactors) where {F, T}
    backend = get_backend(u)
    out = KernelAbstractions.allocate(backend, T, ctx.nt)
    _integral_kernel!(backend)(out, f, eq, u, ctx.shap, ctx.vol,
                               Val(nvariables(eq)); ndrange = ctx.nt)
    KernelAbstractions.synchronize(backend)
    return sum(out)
end

"""
    integrate(u, ctx) -> Vector

Component-wise integrals `∫_Ω u_c dx` of a DG field `u (npl, nc, nt)` — the
conserved totals a scheme should preserve. [`AnalysisCallback`](@ref)
records them at `t0` and reports the drift. Device-capable; returns a host
`Vector` of length `nc`.
"""
function integrate(u::AbstractArray{T, 3}, ctx::GeometricFactors) where {T}
    out = _component_integrals(u, ctx, false)
    return vec(Array(sum(out; dims = 1)))
end

function _component_integrals(u::AbstractArray{T, 3}, ctx, squared::Bool) where {T}
    backend = get_backend(u)
    nc = size(u, 2)
    out = KernelAbstractions.allocate(backend, T, ctx.nt, nc)
    _component_integral_kernel!(backend)(out, u, ctx.shap, ctx.vol, squared;
                                         ndrange = (ctx.nt, nc))
    KernelAbstractions.synchronize(backend)
    return out
end

"""
    l2norm(u, ctx; component=nothing) -> Real

`‖u‖_{L²(Ω)}` of one solution component (or, with `component = nothing`, the
root-sum-square over all components) of a DG field `u (npl, nc, nt)`,
integrated with the quadrature of `ctx::GeometricFactors`. Device-capable.
"""
function l2norm(u::AbstractArray{T, 3}, ctx::GeometricFactors;
                component::Union{Nothing, Integer} = nothing) where {T}
    out = _component_integrals(u, ctx, true)
    sq = component === nothing ? sum(out) : sum(@view out[:, component])
    return sqrt(sq)
end

# L2 error of one component against `exact(x::SVector{Dim}, t)`, on the
# context's quadrature — the AnalysisCallback backend. Since Master(mesh)
# defaults to degree-4p rules this matches the post-hoc l2error to round-off.
function _l2error(exact::F, u::AbstractArray{T, 3}, ctx::GeometricFactors;
                  component::Integer = 1, t::Real = 0.0) where {F, T}
    backend = get_backend(u)
    out = KernelAbstractions.allocate(backend, T, ctx.nt)
    _l2error_kernel!(backend)(out, exact, u, Int(component), T(t), ctx.shap,
                              ctx.vol, Val(ndims(ctx)); ndrange = ctx.nt)
    KernelAbstractions.synchronize(backend)
    return sqrt(sum(out))
end

# max over all solution nodes of the pointwise characteristic-speed bound
# `wavespeed(eq, u)` — the λ of the CFL formula, evaluated on the device
function _max_wavespeed(eq::AbstractEquation, u::AbstractArray{T, 3}) where {T}
    backend = get_backend(u)
    out = KernelAbstractions.allocate(backend, T, size(u, 3))
    _wavespeed_kernel!(backend)(out, eq, u, Val(nvariables(eq));
                                ndrange = size(u, 3))
    KernelAbstractions.synchronize(backend)
    return maximum(out)
end
