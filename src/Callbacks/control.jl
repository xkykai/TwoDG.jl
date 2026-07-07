"""
    StepsizeCallback(; cfl=0.3, schedule=EveryStep(), interval=nothing)

Adaptive CFL time step (Trixi's `StepsizeCallback`): recomputes

    dt = cfl / ( λ (2p+1)/h + κ ((2p+1)/h)² )

from the **current** solution and writes it to `state.dt`, which the RK4
loop reads before every step. `λ` is the maximum of the pointwise
characteristic-speed bound [`wavespeed`](@ref)`(eq, u)` over all solution
nodes (a device reduction), `κ` the equation's [`diffusivity`](@ref), and
`h` the smallest inscribed-simplex diameter of the mesh (cached at
`initialize!`) — the same formula as `compute_dt`, re-evaluated as the
solution develops.

The initial `dt` is set at `initialize!`, so the `dt` passed to `solve` is
only a placeholder. Solve with `tfinal` (the loop then clamps the last step
to land on `tfinal` exactly); with `nstep`, the run simply takes `nstep`
CFL-sized steps. This is the one callback that *controls* the loop rather
than observing it, and it does not translate to the SciML bridge —
OrdinaryDiffEq owns adaptivity there.
"""
mutable struct StepsizeCallback{S <: AbstractSchedule}
    const schedule :: S
    const cfl      :: Float64
    hmin           :: Float64
end

StepsizeCallback(; cfl::Real = 0.3, schedule::AbstractSchedule = EveryStep(),
                 interval = nothing) =
    StepsizeCallback(_schedule(schedule, interval), Float64(cfl), NaN)

function initialize!(cb::StepsizeCallback, state)
    initialize!(cb.schedule, state)
    state.dt = _cfl_dt(cb, state)
    return nothing
end

function (cb::StepsizeCallback)(state)
    cb.schedule(state) || return nothing
    state.dt = _cfl_dt(cb, state)
    return nothing
end

function _cfl_dt(cb::StepsizeCallback, state)
    isnan(cb.hmin) && (cb.hmin = min_inscribed_diameter(state.prob.mesh))
    eq = _equation(state)
    λ = Float64(_max_wavespeed(eq, state.u))
    κ = Float64(diffusivity(eq))
    pfac = (2 * state.prob.mesh.porder + 1) / cb.hmin
    denom = λ * pfac + κ * pfac^2
    denom > 0 ||
        throw(ArgumentError("equation has neither a propagation speed nor a diffusivity"))
    return cb.cfl / denom
end

"""
    NaNCheckCallback(; schedule=EveryStep(), interval=nothing, io=stdout)

Early-abort guard for diverging runs (in the spirit of Oceananigans'
`NaNChecker`): at each firing checks the live solution for non-finite
values (`NaN`/`Inf`) with a single device reduction and, if any are found,
reports the step, the time, and the offending solution component(s) to
`io`, then returns `true` to stop the solve. The partial solution is still
returned, so the incipient blow-up can be inspected (`sol.u`, plotting,
`save_vtk`) instead of burning the rest of the wall-clock budget on a field
of NaNs.

The healthy-path cost is one `isfinite` reduction over the field per
firing; on a GPU each firing synchronizes the device, so pass
`interval = n` to check every `n` steps if that matters.
"""
struct NaNCheckCallback{S <: AbstractSchedule, IOT <: IO}
    schedule :: S
    io       :: IOT
end

NaNCheckCallback(; schedule::AbstractSchedule = EveryStep(),
                 interval = nothing, io::IO = stdout) =
    NaNCheckCallback(_schedule(schedule, interval), io)

function initialize!(cb::NaNCheckCallback, state)
    initialize!(cb.schedule, state)
    return nothing
end

function (cb::NaNCheckCallback)(state)
    cb.schedule(state) || return false
    u = state.u
    all(isfinite, u) && return false
    # failure path only: name the components that went bad (host work and
    # per-component reductions are fine here — the run is being aborted)
    vars = state.prob === nothing ?
           [string("u", c) for c in axes(u, 2)] :
           [string(v) for v in varnames(_equation(state))]
    bad = [vars[c] for c in axes(u, 2)
           if !all(isfinite, @view u[:, c, :])]
    @printf(cb.io, "NaN/Inf at step %d, t = %g in component(s) %s — stopping the solve\n",
            state.step, state.t, join(bad, ", "))
    return true
end
