# TwoDG.jl — Agent Rules

Guidance for AI agents (and humans) working in this repository. It encodes what
*good* high-order PDE-solver code should look like here — treat it as the target,
not merely a description of the current state.

## Project Overview

TwoDG.jl is a Julia framework for solving 2D PDEs with high-order Galerkin
methods — **continuous (CG)**, **discontinuous (DG/LDG)**, and **hybridizable
discontinuous (HDG)** — behind a single `solve` entry point, on CPU or GPU from
the same code. Domains span linear scalar transport, convection–diffusion, the
first-order wave system, compressible Euler, and steady/unsteady incompressible
Navier–Stokes (with Boussinesq buoyancy and superconvergent H(div) postprocessing).

The scientific promise of this package is **provable accuracy**: high-order
methods on curved simplex elements that hit their design convergence rates
(`k+1`, and `k+2` superconvergence for HDG postprocessing). Every change is
judged first by whether it preserves that promise.

## Language & Environment

- **Julia 1.10+** (LTS floor); CI runs 1.10, latest stable, and `pre` on Linux + Windows.
- **CPU and GPU** through [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl):
  the same kernels run on CPU threads or a CUDA device by passing `ArrayT = CuArray`.
- **Key deps**: KernelAbstractions, Krylov, StaticArrays, Adapt, Atomix, ForwardDiff.
- **Optional integrations are package extensions** (`weakdeps` + `ext/`), never
  hard dependencies: `Gmsh` (NACA meshes), `Makie` (plotting), `SciMLBase`
  (`semidiscretize` → OrdinaryDiffEq).

## Architecture

Source is organized as nested submodules under `src/`, each `include`d and
`using`-ed from `src/TwoDG.jl`:

```
src/
├── TwoDG.jl                        # main module: single export list + includes
├── Utils/                          # initu, interpolate, unique_rows, rootfinding
├── Meshes/                         # mesh generation, MeshGeometry + discretize, connectivity
├── Masters/                        # master element, Koornwinder basis, Gauss quadrature
├── Geometry/                       # geometric factors on (curved) elements
├── Drivers/                        # problem-specific drivers (areacircle, trefftz)
├── Plotting/                       # plot API (implementations live in ext/TwoDGMakieExt.jl)
├── ContinuousGalerkin/             # CG assembly, solve, batched path, l2 error
├── DiscontinuousGalerkin/          # explicit (L)DG: rinvexpl/rldgexpl, RK4, KA kernels
├── Apps/                           # equation "apps": flux + numerical flux + BC evaluators
├── HybridizableDiscontinuousGalerkin/  # local solves, static condensation, postprocess, NS
└── Interface/                      # high-level Problem + solve API (the user-facing layer)
```

**Two layers, and the boundary between them matters:**

1. **Low-level solvers** (`ContinuousGalerkin`, `DiscontinuousGalerkin`,
   `HybridizableDiscontinuousGalerkin`, `Apps`) — validated numerical kernels.
   Performance-critical, GPU-aware, tested against known convergence rates.
2. **Interface** (`src/Interface/Interface.jl`) — thin, allocation-light
   `Problem` types (`DGProblem`, `HDGProblem`, `CGProblem`), equation types
   (`ConvectionEquation`, `EulerEquations`, `PoissonEquation`, …), boundary
   conditions (`Dirichlet`, `Neumann`, `SlipWall`, `FarField`, …), and
   algorithm selectors (`RK4`, `Direct`, `GMRES`, `ConjugateGradient`). This
   layer *lowers* friendly objects to the compiled representation the kernels
   expect (e.g. BC objects → `bcm::Vector{Int}` + `bcs::Matrix`). Keep it thin:
   new physics belongs in the solver layer; Interface only wires it up.

New user-facing names must be added to the single `export` block in
`src/TwoDG.jl`.

## Critical Rules for Numerical Code

### Parametric element type — never hardcode `Float64`

The element type `T` is parametric so the whole pipeline can run in `Float32`
(including on GPU). In generic solver/kernel code:

- Never write bare `0.0` / `1.0` / `2.0` where a value of the working precision
  is needed. Use `zero(T)`, `one(T)`, `convert(T, …)`, or derive `T` from the
  data (`eltype(u)`, `eltype(p)`).
- Exactly-representable constants may use the typed literal form (`0.5f0` is fine
  only when you truly mean `Float32`); otherwise stay generic.
- A `Float64` literal silently promotes an entire array and breaks single-precision
  and GPU runs — this is the single most common accuracy/performance regression.

### GPU compatibility (KernelAbstractions)

- Kernels must be **type-stable** and **allocation-free**. Type instability or
  heap allocation in an inner kernel destroys GPU performance.
- Iterate the mesh through the launch/kernel machinery; don't hand-roll element
  loops in a way that can't run on the device.
- Prefer `ifelse` / branch-free arithmetic over data-dependent `if` in the
  hottest kernels.
- Move data with `Adapt.adapt` / the `ArrayT` abstraction — never sprinkle
  literal `Array()` / `CuArray()` conversions through solver code.
- The same source must serve CPU and GPU; do not fork a `_cpu` and `_gpu` copy
  of a kernel.

### Optional backends live in extensions

Anything that pulls a heavy or optional dependency (plotting, Gmsh meshing,
SciML steppers) goes in `ext/` guarded by a `weakdep`, with a stub/`export` in
`src/`. Do **not** add such packages to `[deps]` in the root `Project.toml`.

### Dispatch over conditionals

Use Julia's multiple dispatch and the type system to select behavior (equation,
boundary condition, algorithm) instead of `if`/`elseif` chains on a tag. This is
already the shape of the `Apps`/`Interface` design; extend it, don't bypass it.

### Dependencies

Do not add, remove, or bump entries in the root `Project.toml` `[deps]` /
`[weakdeps]` unless the task truly requires it — changes ripple into CI, load
time, and downstream compat. Only touch `[compat]` when explicitly asked.

## Naming Conventions

- **Types / constructors**: `PascalCase` — `Mesh`, `Master`, `HDGProblem`, `EulerEquations`.
- **Functions / variables**: `snake_case` — `hdg_solve`, `compute_dt`, `l2error`.
- **Mutating functions** end in `!` — `rk4!`, `rinvexpl!`, `mkmesh_distort!`.
- **Numerical variables** may use readable math notation matching the papers
  (`u`, `q`, `uhat`, `κ`, `Δt`, polynomial degree `porder`/`k`). Pick either a
  math symbol or a full English name — don't invent half-abbreviated hybrids.
- **File names** are lower/snake and track their contents (`hdg_postprocess.jl`,
  `makemesh_circle.jl`).
- Keep keyword-argument names consistent across related `Problem`/equation
  constructors (`bc`, `u0`, `source`, `dt`, `tfinal`, `ArrayT`).

## Verification & Testing

This is a numerics package: **a method is not "done" until its convergence rate
is verified.**

- **Every new equation or discretization ships with a convergence test** that
  refines `h` and/or `p` and asserts the expected rate (`k+1` in the natural
  norm; `k+2` for HDG postprocessed quantities). A green unit test that doesn't
  check a rate does not establish correctness for a solver.
- Existing suites live in `test/` (`test_masters`, `test_meshes`, `test_dg`,
  `test_ka`, `test_cg`, `test_hdg`, `test_interface`, `test_gmsh`) and run from
  `test/runtests.jl`. Add new tests into the matching file, or a new
  `test_*.jl` wired into `runtests.jl`.
- **CPU/GPU parity**: KernelAbstractions paths are exercised in `test_ka`; a new
  kernel needs a test that the device result matches the reference.
- **Golden/reference values**: some tests pin numerical outputs. Only re-pin a
  golden when a flux/scheme change is *deliberate* — a golden that moves
  unexpectedly is a bug signal, not a test to silence.
- **Aqua** quality checks run in `test_aqua` (ambiguities, stale deps, exports);
  keep them passing.
- Keep individual tests fast; heavy validation runs belong in `examples/`, not
  the CI test suite.
- **Type stability**: new hot-path code should be checkable with `@code_warntype`
  / `@inferred`; no allocations in inner loops.

## Documentation

- Public functions and types get a docstring: the signature, a one-line purpose,
  the meaning of arguments, and — for numerical methods — a **reference to the
  paper** the scheme comes from (e.g. Nguyen, Peraire & Cockburn, JCP 2011 for
  HDG). See the docstrings in `src/Interface/Interface.jl` for the house style.
- Comments explain **why** (a stabilization choice, an index convention forced by
  the basis ordering, a numerical-stability guard), not **what** the code
  literally does. Delete commented-out code — git is the history.
- User-facing docs live in `docs/` (Documenter + Literate tutorials); update the
  relevant manual page when you change public behavior. Keep `README.md`
  feature claims honest.

## Git & PR Workflow

- Work on feature branches; keep each PR to a single, clearly-titled concern.
- Update tests and docs in the same PR as the code they cover.
- Don't commit or push unless asked; never bypass CI hooks.
- Before considering a change complete: the full test suite passes, new numerical
  behavior has a convergence check, and no `Project.toml` deps drifted unintentionally.
- A performance PR should post before/after timings; a new-method PR should post
  its convergence table.

## Common Pitfalls

1. **Hardcoded `Float64`** in generic code — breaks `Float32`/GPU. Use `zero(T)` etc.
2. **Type instability / allocation** inside a kernel — silent GPU performance cliff.
3. **Manual `Array()` / `CuArray()`** conversions instead of the `ArrayT`/`adapt` abstraction.
4. **Adding a real dependency** for something that should be a package extension.
5. **New physics stuffed into `Interface`** — put the numerics in the solver
   module and let Interface stay a thin lowering layer.
6. **Forgetting to export** a new public name from `src/TwoDG.jl`.
7. **Claiming correctness without a convergence test** — a passing smoke test is
   not proof of order of accuracy.
8. **Silently re-pinning goldens** to make a test go green.
9. **Curved-element accuracy loss** — isoparametric/curved boundary handling is
   load-bearing for design accuracy; changes near the master element, Koornwinder
   basis, or geometric factors need a curved-geometry convergence check.
10. **`if`-on-a-tag** where dispatch on a type would be cleaner and faster.

## Design Principles

- **Provable accuracy first**: preserve design convergence rates; verify them.
- **One API, three methods**: CG, DG, and HDG share meshes, equations, and BCs so
  the same problem can be solved three ways. Keep that symmetry when extending.
- **Same code, CPU and GPU**: no per-backend forks in `src/`.
- **Dispatch over branching**; **extensions over hard deps**; **thin Interface
  over fat kernels**.
- **Defaults serve the common case**: sensible `bc`, `dt`, `ArrayT = Array`
  defaults so the typical user writes little boilerplate.

## Agent Behavior

- Prioritize numerical correctness and type/GPU stability over cleverness.
- Follow the patterns already in the module you're editing.
- When you add a method, add its convergence test and its export in the same change.
- Reference the governing equations / source papers in docstrings and comments.
- If a change touches the master element, basis, quadrature, or geometric
  factors, treat every solver as potentially affected and lean on the convergence
  suite to catch regressions.
