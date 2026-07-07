# Contributing to TwoDG.jl

Thank you for your interest in contributing! This document covers the
practical workflow; the full design rules live in [AGENTS.md](AGENTS.md),
which is the single source of truth for architecture, numerical conventions,
naming, and testing.

## Getting set up

```sh
git clone https://github.com/xkykai/TwoDG.jl
cd TwoDG.jl
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Julia 1.10 or newer is required. Optional features activate through package
extensions when their dependency is loaded: plotting (`CairoMakie`/`GLMakie`),
NACA meshes and `.msh` import (`Gmsh`), ParaView output (`WriteVTK`),
`semidiscretize` (`SciMLBase`/OrdinaryDiffEq), GPU execution (`CUDA`).

## Running the tests

```sh
# full suite (what CI runs)
julia --project=. -e "using Pkg; Pkg.test()"

# a single test file during iteration (much faster)
julia --project=. -e "using TwoDG, Test; include(\"test/test_hdg.jl\")"
```

## The convergence-test requirement

This is a numerics package: **a method is not done until its convergence rate
is verified.** Every new equation or discretization must ship with a test that
refines `h` and/or `p` and asserts the design rate (`k+1` in the natural norm,
`k+2` for HDG postprocessed quantities). A green unit test that does not check
a rate does not establish correctness for a solver. The existing contract is
catalogued in [test/CONVERGENCE.md](test/CONVERGENCE.md) — extend that table
with your test.

Golden/reference values pinned in tests are a drift alarm: never re-pin one to
make a test pass unless the flux/scheme change was deliberate and reviewed.

## Code rules (the short version)

- **Never hardcode `Float64`** in generic solver/kernel code — the element
  type `T` is parametric so `Float32` and GPU runs keep working. Use
  `zero(T)`, `one(T)`, `convert(T, …)`.
- **Kernels must be type-stable and allocation-free**, and the same source
  must serve CPU and GPU (KernelAbstractions) — no `_cpu`/`_gpu` forks, no
  literal `Array()`/`CuArray()` conversions (use `Adapt.adapt`/`ArrayT`).
- **Dispatch over conditionals**: select behavior (equation, BC, algorithm)
  with types, not `if`-chains on tags.
- **Optional backends are package extensions** (`weakdeps` + `ext/`), never
  hard dependencies. Do not touch `[deps]`/`[compat]` in `Project.toml`
  unless the change truly requires it.
- New public names go in the single `export` block of `src/TwoDG.jl`, with a
  docstring (and, for numerical methods, a reference to the source paper).
- New physics belongs in the solver modules; `src/Interface/` stays a thin
  lowering layer.
- **Comments explain why, not what** — a stabilization choice, an index
  convention, a numerical-stability guard. Code comments must never reference
  planning documents, phases, task IDs, or development sessions; state the
  technical fact self-contained. Delete commented-out code — git is the
  history.

## Pull requests

- Work on a feature branch; keep each PR to a single, clearly-titled concern.
- Update tests, docstrings, and the relevant `docs/` page in the same PR as
  the code they cover, and add a `NEWS.md` entry for user-visible changes.
- A performance PR should post before/after timings; a new-method PR should
  post its convergence table.
- CI must be green (tests on Julia 1.10/stable/pre × Linux/Windows, plus Aqua
  quality checks); never bypass hooks.

## Reporting bugs / asking questions

Use the GitHub issue templates. For a numerical-accuracy bug, the most
valuable thing you can include is a minimal script with a manufactured
solution and the observed vs expected convergence table.
