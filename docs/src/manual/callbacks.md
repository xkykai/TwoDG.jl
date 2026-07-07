# Callbacks and diagnostics

The internal RK4 time loop of `solve(prob::DGProblem, RK4(); ...)` accepts a
`callback` — a composable run-time layer for monitoring, analysis, output,
and control while the solve is running, in the spirit of Trixi.jl's callback
set and Oceananigans.jl's schedule/callback split.

Three orthogonal pieces:

- a **schedule** decides *when* to fire ([`EveryStep`](@ref),
  [`IterationInterval`](@ref), [`TimeInterval`](@ref),
  [`SpecifiedTimes`](@ref), [`WallTimeInterval`](@ref)),
- a **callback** decides *what* to do — a plain closure or one of the
  built-ins below,
- a [`SolveState`](@ref) is the documented view of the running solve the
  callback acts on.

```julia
using TwoDG

mesh = mkmesh_square(17, 17, 3, 0, 1)
eq   = EulerEquations(γ = 1.4)
prob = DGProblem(eq, mesh; bc = fill(FarField(uinf), 4), u0)

acb = AnalysisCallback(interval = 50,
                       integrals = (ek = energy_kinetic, s = entropy))
sol = solve(prob, RK4(); dt = compute_dt(prob), tfinal = 1.0,
            callback = CallbackSet(
                ProgressCallback(interval = 100),
                acb,
                SaveSolutionCallback(schedule = TimeInterval(0.1),
                                     fields = (:u, :p => pressure),
                                     path = "out"),
            ))

acb.time, acb.data      # analysis history (also on sol.callbacks)
```

## The calling convention

Any callable `cb(state::SolveState) -> Union{Nothing, Bool}` is a valid
callback; returning `true` stops the solve early. A plain closure works:

```julia
sol = solve(prob, RK4(); dt, nstep = 1000,
            callback = state -> begin
                state.step % 100 == 0 && println("t = ", state.t)
                false
            end)
```

`state.u` is the **live** solution array — device-resident when solving with
`ArrayT = CuArray`; copy it (`Array(state.u)`) if you keep a reference.
[`CallbackSet`](@ref) composes several callbacks: all members run each step,
and the solve stops if any returned `true`.

Callbacks are **observers**: with a fixed `dt`, the computed solution is
bit-identical with and without callbacks attached (the test suite asserts
this). The one sanctioned exception is [`StepsizeCallback`](@ref), which
writes `state.dt` to control the loop.

## The catalog

| Callback | Purpose |
|---|---|
| [`ProgressCallback`](@ref) | heartbeat: step, `t`, `dt`, ms/step, ETA |
| [`AnalysisCallback`](@ref) | conservation drift, min/max, user integral functionals, L² errors; keeps a history |
| [`SteadyStateCallback`](@ref) | stop when the solution stops changing |
| [`NaNCheckCallback`](@ref) | abort a diverging run at the first non-finite value, naming the step and component |
| [`SaveSolutionCallback`](@ref) | solution/derived-field snapshots to disk |
| [`CheckpointCallback`](@ref) | atomic restart files; resume with `solve(...; restart = path)` |
| [`StepsizeCallback`](@ref) | recompute the CFL-limited `dt` from the running solution |

New behaviors should usually be a user closure composed with a schedule, not
a new callback type; anything that can be written as a pointwise functional
`(eq, u::SVector) -> Real` already works with [`integrate`](@ref) and
`AnalysisCallback`'s `integrals`.

## Quadrature-exact diagnostics

[`integrate`](@ref) evaluates `∫ f(eq, u(x)) dx` over the mesh with the
geometry cache's quadrature, as a KernelAbstractions kernel plus device
reduction — on the GPU only the scalar crosses the bus. The pointwise
functional `f` has the same contract as [`derived_field`](@ref), so
[`pressure`](@ref), [`mach`](@ref), [`entropy`](@ref),
[`energy_kinetic`](@ref), [`energy_internal`](@ref),
[`energy_total`](@ref), and any user closure all work:

```julia
ctx = DGContext(ReferenceElement(mesh), mesh)
Ek  = integrate(energy_kinetic, eq, u, ctx)
m   = integrate(u, ctx)          # component-wise conserved totals
nrm = l2norm(u, ctx)             # L² norm through the same quadrature
```

## Restarts

```julia
# long run, checkpointed every 10 wall-clock minutes
solve(prob, RK4(); dt, tfinal = 10.0,
      callback = CheckpointCallback(path = "run.jls"))

# after a crash: resume where the checkpoint left off
solve(prob, RK4(); dt, tfinal = 10.0, restart = "run.jls")
```

Checkpoints are written atomically (temporary file + rename), so an
interrupted run never leaves a torn file; a resumed run reproduces the
uninterrupted one to floating-point tolerance.

## Custom callback types

For reusable callbacks with setup/teardown, define a callable struct and
extend the two lifecycle hooks (module-qualified — they are not exported):

```julia
struct MyProbe
    schedule::IterationInterval
    values::Vector{Float64}
end
(p::MyProbe)(state) = p.schedule(state) ? push!(p.values, maximum(state.u)) : nothing
TwoDG.Callbacks.initialize!(p::MyProbe, state) = push!(p.values, maximum(state.u))
```

## SciML users

When time stepping through `semidiscretize(prob, tspan)` and
OrdinaryDiffEq, use **SciML's** callback system (`DiscreteCallback`,
`TerminateSteadyState`, adaptive steppers instead of `StepsizeCallback`) —
TwoDG's callbacks drive only the internal RK4 loop.
