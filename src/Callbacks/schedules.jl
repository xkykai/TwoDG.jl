"""
    AbstractSchedule

A schedule decides *when* a callback fires; the callback decides *what*
happens — the two concerns never mix (Oceananigans' design). A schedule is a
small callable object: `sched(state::SolveState) -> Bool`. Stateful
schedules are anchored at `t0` by `initialize!(sched, state)` and advance
their own next-fire state when they report `true`.

Built-in schedules: [`EveryStep`](@ref), [`IterationInterval`](@ref),
[`TimeInterval`](@ref), [`SpecifiedTimes`](@ref),
[`WallTimeInterval`](@ref). All built-in callbacks accept `schedule = ...`,
with `interval = n` as sugar for `IterationInterval(n)`.
"""
abstract type AbstractSchedule end

"`EveryStep()` — fire after every time step."
struct EveryStep <: AbstractSchedule end
(::EveryStep)(state) = true

"""
    IterationInterval(n)

Fire every `n` steps (`step % n == 0`).
"""
struct IterationInterval <: AbstractSchedule
    n :: Int
    function IterationInterval(n::Integer)
        n ≥ 1 || throw(ArgumentError("IterationInterval needs n ≥ 1, got $n"))
        return new(n)
    end
end
(s::IterationInterval)(state) = state.step % s.n == 0

"""
    TimeInterval(Δt)

Fire on the first step whose time reaches the next multiple `t0 + k·Δt`
(fixed-step loop: no interpolation to the exact time — the firing `t` may
overshoot by up to one `dt`). A step that crosses several multiples fires
once.
"""
mutable struct TimeInterval <: AbstractSchedule
    const Δt :: Float64
    next     :: Float64
    function TimeInterval(Δt::Real)
        Δt > 0 || throw(ArgumentError("TimeInterval needs Δt > 0, got $Δt"))
        return new(Float64(Δt), NaN)
    end
end

function initialize!(s::TimeInterval, state)
    s.next = state.t + s.Δt
    return nothing
end

function (s::TimeInterval)(state)
    # unanchored (used without the loop's initialize!): fire now, then align
    isnan(s.next) && (s.next = state.t)
    tol = 1e-10 * s.Δt
    state.t + tol < s.next && return false
    while s.next ≤ state.t + tol
        s.next += s.Δt
    end
    return true
end

"""
    SpecifiedTimes(ts...)
    SpecifiedTimes(ts::AbstractVector)

Fire on the first step whose time reaches each of the given times (sorted
internally). Times beyond the end of the run simply never fire; a step that
crosses several times fires once.
"""
mutable struct SpecifiedTimes <: AbstractSchedule
    const times :: Vector{Float64}
    idx         :: Int
end
SpecifiedTimes(ts::AbstractVector) = SpecifiedTimes(sort!(collect(Float64, ts)), 1)
SpecifiedTimes(ts::Real...) = SpecifiedTimes(collect(Float64, ts))

function initialize!(s::SpecifiedTimes, state)
    s.idx = 1
    while s.idx ≤ length(s.times) && s.times[s.idx] ≤ state.t + _time_tol(s.times[s.idx])
        s.idx += 1
    end
    return nothing
end

_time_tol(t) = 1e-12 * max(abs(t), 1.0)

function (s::SpecifiedTimes)(state)
    fired = false
    while s.idx ≤ length(s.times) && state.t ≥ s.times[s.idx] - _time_tol(s.times[s.idx])
        s.idx += 1
        fired = true
    end
    return fired
end

"""
    WallTimeInterval(seconds)

Fire when at least `seconds` of wall-clock time have elapsed since the last
firing (or since `initialize!`). For heartbeats and
[`CheckpointCallback`](@ref)s on long runs.
"""
mutable struct WallTimeInterval <: AbstractSchedule
    const seconds :: Float64
    last          :: UInt64
    function WallTimeInterval(seconds::Real)
        seconds ≥ 0 || throw(ArgumentError("WallTimeInterval needs seconds ≥ 0, got $seconds"))
        return new(Float64(seconds), UInt64(0))
    end
end

function initialize!(s::WallTimeInterval, state)
    s.last = time_ns()
    return nothing
end

function (s::WallTimeInterval)(state)
    s.last == 0 && (s.last = time_ns())
    (time_ns() - s.last) / 1e9 ≥ s.seconds || return false
    s.last = time_ns()
    return true
end

# `interval = n` sugar shared by every built-in callback constructor
_schedule(schedule::AbstractSchedule, interval::Nothing) = schedule
_schedule(schedule::AbstractSchedule, interval::Integer) = IterationInterval(interval)
