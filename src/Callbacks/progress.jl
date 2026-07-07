"""
    ProgressCallback(; schedule=IterationInterval(100), interval=nothing, io=stdout)

Heartbeat for long runs (Trixi's `AliveCallback`): at each firing prints one
line with the step, time, current `dt`, wall-clock rate since the last
firing, and an ETA (from `nstep` or `tfinal`, whichever drives the run); at
the end of the solve prints the total wall time. Never touches the
solution; allocation-free on steps where it does not fire.
"""
mutable struct ProgressCallback{S <: AbstractSchedule, IOT <: IO}
    const schedule :: S
    const io       :: IOT
    start_ns       :: UInt64
    last_ns        :: UInt64
    last_step      :: Int
end

ProgressCallback(; schedule::AbstractSchedule = IterationInterval(100),
                 interval = nothing, io::IO = stdout) =
    ProgressCallback(_schedule(schedule, interval), io, UInt64(0), UInt64(0), 0)

function initialize!(cb::ProgressCallback, state)
    initialize!(cb.schedule, state)
    cb.start_ns = cb.last_ns = time_ns()
    cb.last_step = state.step
    return nothing
end

function (cb::ProgressCallback)(state)
    cb.schedule(state) || return nothing
    now = time_ns()
    cb.start_ns == 0 && (cb.start_ns = now)   # used without initialize!
    nsteps = state.step - cb.last_step
    rate_ms = (cb.last_ns == 0 || nsteps ≤ 0) ? NaN :
              (now - cb.last_ns) / 1e6 / nsteps

    @printf(cb.io, "step %d", state.step)
    state.nstep < typemax(Int) && @printf(cb.io, "/%d", state.nstep)
    @printf(cb.io, "  t = %-11.5g dt = %-10.4g", state.t, state.dt)
    isnan(rate_ms) || @printf(cb.io, " %9.3g ms/step", rate_ms)
    eta = _eta_seconds(state, rate_ms)
    isnan(eta) || print(cb.io, "  ETA ", _fmt_seconds(eta))
    println(cb.io)

    cb.last_ns = now
    cb.last_step = state.step
    return nothing
end

function finish!(cb::ProgressCallback, state)
    cb.start_ns == 0 && return nothing
    elapsed = (time_ns() - cb.start_ns) / 1e9
    @printf(cb.io, "finished at step %d, t = %.5g (%s wall time)\n",
            state.step, state.t, _fmt_seconds(elapsed))
    return nothing
end

function _eta_seconds(state, rate_ms)
    isnan(rate_ms) && return NaN
    remaining = if state.nstep < typemax(Int)
        Float64(state.nstep - state.step)
    elseif !isnan(state.tfinal) && state.dt > 0
        (state.tfinal - state.t) / state.dt
    else
        return NaN
    end
    return max(remaining, 0.0) * rate_ms / 1e3
end

function _fmt_seconds(s::Real)
    s < 60 && return @sprintf("%.3g s", s)
    s < 3600 && return @sprintf("%dm%02ds", s ÷ 60, round(Int, s % 60))
    return @sprintf("%dh%02dm", s ÷ 3600, round(Int, (s % 3600) ÷ 60))
end
