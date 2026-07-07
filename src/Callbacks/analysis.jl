"""
    AnalysisCallback(; schedule=IterationInterval(100), interval=nothing,
                     integrals=NamedTuple(), errors=nothing, io=stdout)

In-loop solution analysis (Trixi's `AnalysisCallback`): at `t0`, at each
firing of `schedule`, and at the end of the run, computes

- the component-wise conserved integrals `∫ u_c dx` and their maximum drift
  against the `t0` values (a conservative scheme should keep this at
  round-off),
- the per-component min/max of the nodal values,
- the user `integrals` — a `NamedTuple` of pointwise functionals
  `(eq, u::SVector) -> Real` (e.g. `(ek = energy_kinetic, s = entropy)` or
  any closure), each integrated with [`integrate`](@ref),
- optionally the L² errors against `errors` — a function or tuple of
  functions `exact(x::SVector{Dim}, t) -> Real`, one per solution component
  starting from the first, integrated with the context's quadrature,

prints them as a table row to `io`, and appends them to the history carried
by the callback: `cb.time::Vector{Float64}`, `cb.steps::Vector{Int}`, and
`cb.data::Dict{Symbol, Vector{Float64}}` with keys `:conservation_drift`,
`Symbol(:integral_, var)`, `Symbol(:min_, var)` / `Symbol(:max_, var)` per
component name, each user-integral name, and `Symbol(:l2error_, var)` per
error function. The history also rides on the returned solution as
`sol.callbacks`.

All reductions run on the device holding `state.u`; the callback is an
observer — the computed solution is bit-identical with and without it.
"""
mutable struct AnalysisCallback{S <: AbstractSchedule, I <: NamedTuple, E,
                                IOT <: IO}
    const schedule  :: S
    const integrals :: I
    const errors    :: E
    const io        :: IOT
    const u0_totals :: Vector{Float64}
    const time      :: Vector{Float64}
    const steps     :: Vector{Int}
    const data      :: Dict{Symbol, Vector{Float64}}
end

function AnalysisCallback(; schedule::AbstractSchedule = IterationInterval(100),
                          interval = nothing, integrals = NamedTuple(),
                          errors = nothing, io::IO = stdout)
    errs = errors === nothing ? nothing :
           errors isa Tuple ? errors : (errors,)
    return AnalysisCallback(_schedule(schedule, interval),
                            _named_functionals(integrals), errs, io,
                            Float64[], Float64[], Int[],
                            Dict{Symbol, Vector{Float64}}())
end

_named_functionals(nt::NamedTuple) = nt
_named_functionals(fs::Tuple) =
    NamedTuple{ntuple(i -> Symbol(nameof(typeof(fs[i]))), length(fs))}(fs)
_named_functionals(f) = _named_functionals((f,))

function initialize!(cb::AnalysisCallback, state)
    initialize!(cb.schedule, state)
    _print_header(cb, state)
    _analyze!(cb, state)
    return nothing
end

function (cb::AnalysisCallback)(state)
    cb.schedule(state) || return nothing
    _analyze!(cb, state)
    return nothing
end

function finish!(cb::AnalysisCallback, state)
    # make sure the final state is on record even if the schedule missed it
    (isempty(cb.steps) || cb.steps[end] != state.step) && _analyze!(cb, state)
    return nothing
end

_push!(d::Dict{Symbol, Vector{Float64}}, k::Symbol, v) =
    push!(get!(() -> Float64[], d, k), Float64(v))

function _analyze!(cb::AnalysisCallback, state)
    eq = _equation(state)
    u, ctx = state.u, state.ctx
    vars = varnames(eq)

    totals = Float64.(integrate(u, ctx))
    isempty(cb.u0_totals) && append!(cb.u0_totals, totals)
    drift = maximum(abs.(totals .- cb.u0_totals))

    push!(cb.time, state.t)
    push!(cb.steps, state.step)
    _push!(cb.data, :conservation_drift, drift)

    row = Float64[drift]
    for (c, var) in enumerate(vars)
        uc = @view u[:, c, :]
        umin, umax = Float64(minimum(uc)), Float64(maximum(uc))
        _push!(cb.data, Symbol(:integral_, var), totals[c])
        _push!(cb.data, Symbol(:min_, var), umin)
        _push!(cb.data, Symbol(:max_, var), umax)
        push!(row, umin, umax)
    end
    for (name, f) in pairs(cb.integrals)
        val = Float64(integrate(f, eq, u, ctx))
        _push!(cb.data, name, val)
        push!(row, val)
    end
    if cb.errors !== nothing
        for (c, exact) in enumerate(cb.errors)
            err = Float64(_l2error(exact, u, ctx; component = c, t = state.t))
            _push!(cb.data, Symbol(:l2error_, vars[c]), err)
            push!(row, err)
        end
    end

    @printf(cb.io, "%8d  %-12.6g", state.step, state.t)
    foreach(v -> @printf(cb.io, " %-12.5g", v), row)
    println(cb.io)
    return nothing
end

function _print_header(cb::AnalysisCallback, state)
    eq = _equation(state)
    vars = varnames(eq)
    cols = ["|Δ∫u|"]
    for var in vars
        push!(cols, "min($var)", "max($var)")
    end
    append!(cols, string.(keys(cb.integrals)))
    cb.errors === nothing ||
        append!(cols, ["L2err($(vars[c]))" for c in eachindex(cb.errors)])
    println(cb.io, "─"^(22 + 13 * length(cols)))
    @printf(cb.io, "%8s  %-12s", "step", "t")
    foreach(c -> @printf(cb.io, " %-12s", c), cols)
    println(cb.io)
    println(cb.io, "─"^(22 + 13 * length(cols)))
    return nothing
end

"""
    SteadyStateCallback(; abstol=1e-8, reltol=1e-6,
                        schedule=IterationInterval(10), interval=nothing)

Terminate the solve when the solution stops changing: at each firing
compares the finite-difference rate `‖u - u_prev‖₂ / Δt` between consecutive
firings against `abstol + reltol * ‖u‖₂` and returns `true` (stop) once it
falls below. The RK4 stepper does not expose its stage residuals, so the
criterion is this between-firings rate — fire it every few steps, not every
step, for a meaningful Δt. The comparison state is kept on the same device
as `u`; nothing crosses the bus but scalars.
"""
mutable struct SteadyStateCallback{S <: AbstractSchedule}
    const schedule :: S
    const abstol   :: Float64
    const reltol   :: Float64
    uprev          :: Any      # device copy of u at the last firing (lazy)
    tprev          :: Float64
end

SteadyStateCallback(; abstol::Real = 1e-8, reltol::Real = 1e-6,
                    schedule::AbstractSchedule = IterationInterval(10),
                    interval = nothing) =
    SteadyStateCallback(_schedule(schedule, interval), Float64(abstol),
                        Float64(reltol), nothing, NaN)

function initialize!(cb::SteadyStateCallback, state)
    initialize!(cb.schedule, state)
    cb.uprev = copy(state.u)
    cb.tprev = state.t
    return nothing
end

function (cb::SteadyStateCallback)(state)
    cb.schedule(state) || return false
    if cb.uprev === nothing            # used without initialize!
        cb.uprev = copy(state.u)
        cb.tprev = state.t
        return false
    end
    Δt = state.t - cb.tprev
    Δt > 0 || return false
    rate = sqrt(mapreduce((a, b) -> abs2(a - b), +, state.u, cb.uprev)) / Δt
    unorm = sqrt(mapreduce(abs2, +, state.u))
    copyto!(cb.uprev, state.u)
    cb.tprev = state.t
    return rate < cb.abstol + cb.reltol * unorm
end
