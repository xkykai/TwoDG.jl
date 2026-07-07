# # Callbacks and diagnostics
#
# TwoDG's internal RK4 loop accepts **callbacks** — a composable run-time
# layer for monitoring, analysis, output, and control while a solve runs,
# in the spirit of Trixi.jl's callback set and Oceananigans.jl's
# schedule/callback split. This tutorial drives the whole catalog on one
# problem: a Gaussian bump transported by constant-velocity convection,
# which has the exact solution ``u(\boldsymbol{x}, t) =
# u_0(\boldsymbol{x} - \boldsymbol{v} t)`` — perfect for watching in-loop
# error norms. The concepts are documented in the
# [Callbacks and diagnostics](../manual/callbacks.md) manual page.

using TwoDG
using StaticArrays

v = SVector(1.0, 0.5)
bump(x, y) = exp(-100 * ((x - 0.35)^2 + (y - 0.4)^2))
exact(x, t) = bump(x[1] - v[1] * t, x[2] - v[2] * t)   # SVector-position form

mesh = mkmesh_square(17, 17, 3, 0, 1)
eq   = ConvectionEquation(v)
bc   = (bottom = FarField(SVector(0.0)), right = FarField(SVector(0.0)),
        top    = FarField(SVector(0.0)), left  = FarField(SVector(0.0)))
prob = DGProblem(eq, mesh; bc, u0 = [bump]);

# ## Progress, analysis, and snapshots
#
# Three orthogonal pieces compose: a **schedule** decides *when* to fire
# ([`IterationInterval`](@ref), [`TimeInterval`](@ref), [`EveryStep`](@ref),
# [`SpecifiedTimes`](@ref), [`WallTimeInterval`](@ref)), a **callback**
# decides *what* to do, and a [`CallbackSet`](@ref) runs several in order.
#
# [`AnalysisCallback`](@ref) is the workhorse: it records the conserved
# integrals and their drift, per-component min/max, any user **integral
# functionals** (pointwise `(eq, u::SVector) -> Real` closures, integrated
# with the mesh quadrature), and L² errors against an exact solution
# `exact(x::SVector, t)`. The printing callbacks write to the `io` you give
# them — `stdout` by default; a buffer here, so the run log renders as a
# block below:

log = IOBuffer()
acb = AnalysisCallback(interval = 100,
                       integrals = (u_sq = (eq, u) -> u[1]^2,),
                       errors = exact, io = log)

snapdir = mktempdir()
scb = SaveSolutionCallback(schedule = TimeInterval(0.05),
                           path = snapdir, fields = (:u,))

sol = solve(prob, RK4(); dt = 5e-4, tfinal = 0.2,
            callback = CallbackSet(ProgressCallback(interval = 200, io = log),
                                   acb, scb));

# The run log — progress heartbeats interleaved with the analysis table:

print(String(take!(log)))

# The analysis history stays on the callback (and rides along as
# `sol.callbacks`): the conservation drift is at round-off, and the in-loop
# L² error grows only through the dissipation of the scheme itself:

acb.time, acb.data[:l2error_u]

#-

acb.data[:conservation_drift]

# The snapshots are plain `Serialization` files, one `NamedTuple` per
# firing — [`SaveSolutionCallback`](@ref) can also store derived fields via
# `fields = (:u, :p => pressure)`-style pairs:

using Serialization
snap = deserialize(first(scb.files))
snap.t, size(snap.u)

# Callbacks are **observers**: with a fixed `dt` the computed solution is
# bit-identical with and without them (the test suite asserts this), so you
# never trade diagnostics against reproducibility.
#
# A bare closure is also a valid callback — return `true` to stop the solve
# early:
#
# ```julia
# solve(prob, RK4(); dt, tfinal,
#       callback = state -> maximum(abs, state.u) > 10)   # blow-up guard
# ```

# ## Checkpoint and restart
#
# [`CheckpointCallback`](@ref) atomically serializes the full solver state
# (by wall-clock schedule in production; by iteration count here so the
# example is deterministic). `solve(...; restart = path)` resumes where the
# checkpoint left off:

chk = joinpath(snapdir, "checkpoint.jls")
solve(prob, RK4(); dt = 5e-4, tfinal = 0.1,
      callback = CheckpointCallback(path = chk, schedule = IterationInterval(100)));

sol_resumed = solve(prob, RK4(); dt = 5e-4, tfinal = 0.2, restart = chk)
maximum(abs, sol_resumed.u .- sol.u)   # matches the uninterrupted run to FP tolerance

# ## CFL-driven step control
#
# [`StepsizeCallback`](@ref) is the one callback that *controls* the loop
# instead of observing it: every firing recomputes the CFL-limited `dt`
# from the current solution's [`wavespeed`](@ref) and writes it to
# `state.dt` (the `dt` passed to `solve` is only a placeholder):

nsteps = Ref(0)
sol_cfl = solve(prob, RK4(); dt = 1.0, tfinal = 0.2,
                callback = CallbackSet(StepsizeCallback(cfl = 0.3),
                                       state -> (nsteps[] += 1; false)))
nsteps[]

# ## Quadrature-exact diagnostics, post hoc
#
# The same [`integrate`](@ref)/[`l2norm`](@ref) primitives the
# `AnalysisCallback` uses work outside the loop — any pointwise functional
# with the [`derived_field`](@ref) contract integrates over the mesh with
# the geometry cache's quadrature (on GPUs, only the scalar leaves the
# device):

ctx = DGContext(ReferenceElement(mesh), mesh)
integrate((eq, u) -> u[1]^2, eq, sol.u, ctx), l2norm(sol.u, ctx)

# ## Notes
#
# - `state.u` is the **live** solution array — device-resident under
#   `ArrayT = CuArray`; copy it (`Array(state.u)`) before keeping a
#   reference or doing I/O. The built-in output callbacks do this for you.
# - [`SteadyStateCallback`](@ref) stops a run when the finite-difference
#   rate `‖Δu‖/Δt` stalls — the catalog's sixth member, for
#   steady-state-seeking runs.
# - With `semidiscretize` + OrdinaryDiffEq, use **SciML's** callback system
#   instead; TwoDG's callbacks drive only the internal RK4 loop.
