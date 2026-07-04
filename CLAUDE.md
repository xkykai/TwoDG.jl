# CLAUDE.md

The full project guidance lives in **[AGENTS.md](AGENTS.md)** — read it first.
It is the single source of truth for architecture, numerical rules, naming,
testing, and design principles. This file only adds Claude Code / local-machine
specifics.

@AGENTS.md

## Running things locally

This repo is developed on **Windows with PowerShell** as well as Linux CI, so
prefer commands that work in both.

- **Run the test suite:**
  ```
  julia --project=. -e "using Pkg; Pkg.test()"
  ```
- **Run a single test file** during iteration (faster than the whole suite):
  ```
  julia --project=. -e "using TwoDG, Test; include(\"test/test_hdg.jl\")"
  ```
- **Run an example / validation script:**
  ```
  julia --project=. examples/hdg/runhdg_poisson.jl
  ```
- **GPU runs** need `using CUDA` and `ArrayT = CuArray`; **plotting** needs a
  Makie backend (`using CairoMakie`); **`semidiscretize`** needs SciMLBase /
  OrdinaryDiffEq loaded — these are package extensions, so the feature only
  activates once the weakdep is loaded.

## Quick reminders (see AGENTS.md for the full rules)

- Never hardcode `Float64` in generic solver/kernel code — the element type `T`
  is parametric (`Float32`/GPU must keep working). Use `zero(T)`, `one(T)`,
  `convert(T, …)`.
- A new equation or discretization is not done until a **convergence test**
  proves its design rate (`k+1`, or `k+2` for HDG postprocessing).
- Optional backends (Gmsh, Makie, SciML) stay in `ext/` as package extensions —
  don't move them into `[deps]`.
- Export new public names from the single `export` block in `src/TwoDG.jl`.
- Don't commit, push, or re-pin golden test values unless explicitly asked.
