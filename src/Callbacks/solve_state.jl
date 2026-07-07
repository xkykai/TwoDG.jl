"""
    SolveState

The view of a running solve that a callback receives after every step (and
at `t0` through [`initialize!`](@ref)). Fields:

- `u` — the **live** solution array `(npl, nc, nt)`. Under `ArrayT = CuArray`
  it is device-resident; copy it (`Array(state.u)`) if you keep a reference.
- `t::Float64`, `step::Int` — current time and step count.
- `dt::Float64` — the time step. Writable: a callback that assigns
  `state.dt` (see [`StepsizeCallback`](@ref)) controls the loop from the
  next step on.
- `nstep::Int`, `tfinal::Float64` — the planned end of the run
  (`typemax(Int)` / `NaN` when the run is driven by the other criterion);
  used for progress/ETA reporting.
- `prob`, `ctx`, `phys` — the problem, the `GeometricFactors` geometry cache
  (for quadrature-exact diagnostics, see [`integrate`](@ref)), and the
  compiled physics bundle.

These fields are the documented callback API; anything else is internal.
"""
mutable struct SolveState{U, P, C, PH}
    const u      :: U
    t            :: Float64
    step         :: Int
    dt           :: Float64
    const nstep  :: Int
    const tfinal :: Float64
    const prob   :: P
    const ctx    :: C
    const phys   :: PH
end

Base.show(io::IO, s::SolveState) =
    print(io, "SolveState(step ", s.step, ", t = ", s.t, ", dt = ", s.dt, ")")

# the equation behind a running solve (duck-typed so hand-built states with
# `prob = nothing` can still drive physics-free callbacks in tests)
_equation(state::SolveState) = state.prob.equation

"""
    initialize!(cb, state) -> nothing

Lifecycle hook called once by the solve loop before the first step, with
`state.t == t0`. The default is a no-op, so plain closures need nothing;
built-in callbacks use it to anchor time-based schedules, record the `t0`
conservation reference, print headers, etc. Extend it for custom callback
types (`TwoDG.Callbacks.initialize!(cb::MyCallback, state) = ...`).
"""
initialize!(cb, state) = nothing

"""
    finish!(cb, state) -> nothing

Lifecycle hook called once by the solve loop after the last step (regular
end or callback-requested stop). No-op by default; built-ins use it for a
final analysis row / snapshot / timing summary.
"""
finish!(cb, state) = nothing

"""
    CallbackSet(callbacks...)

Compose callbacks: calling the set calls **every** member in order (all run
even if an earlier one requests a stop) and returns `true` — stop the solve
— if any member returned `true`. `initialize!`/`finish!` forward to every
member. Any mix of built-in callbacks and plain closures is allowed.
"""
struct CallbackSet{T <: Tuple}
    callbacks :: T
end
CallbackSet(cbs...) = CallbackSet(cbs)

(cs::CallbackSet)(state) = _run_all(cs.callbacks, state)

# recursive tuple walk: type-stable, allocation-free, and `|` (not `||`) so
# every callback runs before the stop decision is taken
_run_all(::Tuple{}, state) = false
_run_all(cbs::Tuple, state) =
    (first(cbs)(state) === true) | _run_all(Base.tail(cbs), state)

function initialize!(cs::CallbackSet, state)
    foreach(cb -> initialize!(cb, state), cs.callbacks)
    return nothing
end

function finish!(cs::CallbackSet, state)
    foreach(cb -> finish!(cb, state), cs.callbacks)
    return nothing
end

Base.show(io::IO, cs::CallbackSet) =
    print(io, "CallbackSet(", join(map(cb -> nameof(typeof(cb)), cs.callbacks), ", "), ")")
