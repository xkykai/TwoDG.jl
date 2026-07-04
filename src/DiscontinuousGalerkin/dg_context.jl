using ..Geometry: GeometricFactors

"""
    DGContext

Alias for [`GeometricFactors`](@ref TwoDG.Geometry.GeometricFactors) — the
one-time geometry/connectivity cache the KernelAbstractions residual path
consumes. Construct with `DGContext(master, mesh; T=Float64)` and move to a
GPU with `Adapt.adapt(CuArray, ctx)`.
"""
const DGContext = GeometricFactors
