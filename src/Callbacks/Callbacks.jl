"""
Composable run-time layer for the internal time-stepping loop, following
Trixi.jl's callback catalog and Oceananigans.jl's schedule/callback split
(CALLBACKS_PLAN.md): *schedules* decide when to fire, *callbacks* decide
what to do, and a [`SolveState`](@ref) is the documented view of the running
solve they act on.

Any callable `cb(state::SolveState) -> Union{Nothing, Bool}` is a valid
callback — a plain closure, one of the built-ins ([`ProgressCallback`](@ref),
[`AnalysisCallback`](@ref), [`SteadyStateCallback`](@ref),
[`SaveSolutionCallback`](@ref), [`CheckpointCallback`](@ref),
[`StepsizeCallback`](@ref)), or a [`CallbackSet`](@ref) composing several.
Returning `true` stops the solve. Custom callback types may additionally
extend [`initialize!`](@ref) and [`finish!`](@ref), which the solve loop
calls once before the first step and once after the last.
"""
module Callbacks

using Printf
using Serialization
using StaticArrays
using LinearAlgebra: norm
using KernelAbstractions
using KernelAbstractions: @kernel, @index, get_backend
using ..Equations
using ..Equations: nvariables, varnames, derived_field, wavespeed, diffusivity
using ..Geometry: GeometricFactors, VolumeTables, quad_weight, quad_coords,
                  min_inscribed_diameter

export SolveState, CallbackSet,
       AbstractSchedule, EveryStep, IterationInterval, TimeInterval,
       SpecifiedTimes, WallTimeInterval,
       ProgressCallback, AnalysisCallback, SteadyStateCallback,
       SaveSolutionCallback, CheckpointCallback, StepsizeCallback,
       integrate, l2norm

include("solve_state.jl")
include("schedules.jl")
include("diagnostics.jl")
include("progress.jl")
include("analysis.jl")
include("output.jl")
include("control.jl")

end # module Callbacks
