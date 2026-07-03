# TwoDG.jl

[![Build Status](https://github.com/xkykai/TwoDG.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/xkykai/TwoDG.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/xkykai/TwoDG.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/xkykai/TwoDG.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance Julia framework for solving 2D partial differential equations using advanced finite element methods: high-order (hybridizable) discontinuous Galerkin solvers with a single `solve` entry point, running on CPU or GPU from the same code.

## Installation

TwoDG is not yet registered; install it directly from GitHub:

```julia
julia> ]  # enter Pkg mode
pkg> add https://github.com/xkykai/TwoDG.jl
```

Julia 1.10 or newer is required. Plotting (`scaplot`, `meshplot`) activates when a Makie backend is loaded (`using CairoMakie`); GPU runs activate with `using CUDA` (or another KernelAbstractions backend); `semidiscretize` activates with SciMLBase/OrdinaryDiffEq; NACA meshes with `using Gmsh`.

## Cool Visuals

<p align="center">
  <img src="figures/eulerchannel_machnumber.gif" height="300" />
  <br>
  <em>Compressible flow through a channel with a bump computed with 2D Euler equations showing evolution of Mach number</em>
</p>

<p align="center">
  <img src="figures/cp_trefftz_10.png" height="350" />
  <img src="figures/hdg_convdiff_ustar_size_0.2_k_1_c_10_10_p_4.png" height="350" />
  <br>
  <em>Pressure coefficient of a potential flow solution (left) and convection-diffusion solution on an unstructured mesh with Hybridizable Discontinuous Galerkin (HDG) (right)</em>
</p>

<p align="center">
  <img src="figures/hdg_ns_boussinesq_temperature.png" height="350" />
  <img src="figures/hdg_ns_boussinesq_speed.png" height="350" />
  <br>
  <em>Natural convection in a differentially heated cavity at Ra = 10⁴ (incompressible nonhydrostatic Navier-Stokes equations with the Boussinesq approximation, HDG k = 3): temperature (left) and speed (right). The steady hot-wall Nusselt number matches the de Vahl Davis benchmark.</em>
</p>

<p align="center">
  <img src="figures/hdg_ns_kovasznay_convergence.png" width="1000" />
  <br>
  <em>Verification of the HDG incompressible Navier-Stokes solver with the Kovasznay flow at Re = 20: optimal k+1 convergence of velocity, pressure, and velocity gradient, and k+2 superconvergence of the exactly divergence-free, H(div)-conforming postprocessed velocity u*</em>
</p>


## Overview

TwoDG.jl provides a unified implementation of three powerful discretization schemes for 2D PDEs:

- **Continuous Galerkin (CG)** - Traditional continuous finite elements
- **Discontinuous Galerkin (DG)** - High-order explicit time-stepping methods
- **Hybridizable Discontinuous Galerkin (HDG)** - Efficient implicit solvers with static condensation and parallel capabilities

Whether you're studying wave propagation, compressible flows, or convection-diffusion phenomena, TwoDG.jl offers the numerical tools to tackle these problems with arbitrary polynomial orders and sophisticated error analysis.

## Key Features

- **Multiple PDE Types**: Poisson, convection-diffusion, wave equations, Euler equations, and the incompressible Navier-Stokes equations (with optional Boussinesq buoyancy)
- **High-Order Accuracy**: Arbitrary polynomial order support (p-refinement) with Koornwinder orthogonal basis
- **Efficient HDG**: Static condensation dramatically reduces system size compared to standard DG
- **CPU/GPU Portability**: KernelAbstractions-based DG residuals and batched HDG assembly run unchanged on CPU threads or CUDA GPUs (`ArrayT = CuArray`)
- **Rich Mesh Support**: Built-in generators for squares, circles, L-shapes, NACA airfoils, and more, with named boundary tags
- **SciML Interop**: `semidiscretize(prob, tspan)` hands the DG right-hand side to any OrdinaryDiffEq.jl integrator; `compute_dt` picks a CFL-limited explicit step
- **Comprehensive Examples**: Convergence studies, wave scattering, channel flows, and transport problems

## Supported Features

| | Equations | Time / solver | Backends |
|---|---|---|---|
| **DG / LDG** (explicit) | convection, convection-diffusion (LDG), first-order wave system, compressible Euler (Roe flux) | internal `RK4()`, or any OrdinaryDiffEq stepper via `semidiscretize` | CPU + GPU (KernelAbstractions) |
| **HDG** (implicit, static condensation) | Poisson, steady convection-diffusion | `Direct()` sparse LU, `GMRES()` (Krylov.jl, block-Jacobi preconditioned, batched assembly) | CPU + GPU trace solve |
| **HDG Navier-Stokes** | steady/unsteady incompressible NS, Boussinesq buoyancy; superconvergent H(div) postprocessing | Newton + direct/GMRES (driver-level API, see `examples/hdg/`) | CPU (threaded) |
| **CG** | Poisson, convection-diffusion-reaction | `Direct()` | CPU |

Meshes: structured square/L-shape, unstructured circle (distmesh), cos²-bump duct, Trefftz airfoil (conformal map), NACA 4-digit via Gmsh (package extension). All support curved isoparametric elements at arbitrary `porder`; generators attach named boundary tags (`boundary_names(mesh)`). A `MeshGeometry` + `discretize(geo, porder)` two-stage API separates geometry from discretization.

Element type is parametric (`T = Float32` runs the whole DG loop in single precision, on GPU too). Postprocessing: `l2error`, HDG local postprocessing (`p+2` superconvergence), Makie plotting via extension.

## What Can You Do With It?

- **Run convergence studies** to verify optimal rates across different polynomial orders
- **Compare discretization methods** (CG vs DG vs HDG) on the same problems
- **Simulate wave scattering** on complex geometries with absorption boundaries
- **Solve compressible flow** problems including shock waves in channels
- **Solve incompressible flow** problems (steady or time-dependent Navier-Stokes, natural convection with the Boussinesq approximation) with the HDG method of Nguyen, Peraire & Cockburn (JCP, 2011)
- **Analyze convection-diffusion** transport with various stabilization parameters
- **Develop new numerical methods** using the extensible master element framework

## Quick Example

```julia
using TwoDG

# unit square, 8×8×2 elements of polynomial order 3
mesh = mkmesh_square(9, 9, 3, 0, 1)

# steady Poisson problem, -Δu = f, u = 0 on the boundary, solved with HDG
f(p) = 2π^2 .* sin.(π .* p[:, 1]) .* sin.(π .* p[:, 2])
prob = HDGProblem(PoissonEquation(), mesh; bc = Dirichlet(0.0), source = f)
sol  = solve(prob)                       # or solve(prob, Direct())

l2error(sol, (x, y) -> sin(π * x) * sin(π * y))   # ~1e-6

# plotting needs a Makie backend: `using CairoMakie`, then
# scaplot(mesh, sol.u[:, 1, :], show_mesh = true)
```

Time-dependent conservation laws use `DGProblem` + `RK4()`, with boundary
conditions by name and a CFL-based time step (`ArrayT = CuArray` runs the
whole loop on a GPU):

```julia
eq   = ConvectionDiffusionEquation([1.0, 0.5], 0.01)
prob = DGProblem(eq, mesh;
                 bc = (bottom = Dirichlet(), right = Neumann(),
                       top    = Dirichlet(), left  = Neumann()),
                 u0 = [(x, y) -> exp(-16 * ((x - 0.5)^2 + (y - 0.5)^2))])
sol  = solve(prob, RK4(); dt = compute_dt(prob), tfinal = 1.0)

# or hand the semidiscretization to OrdinaryDiffEq (needs SciMLBase loaded):
using OrdinaryDiffEqTsit5
ode = semidiscretize(prob, (0.0, 1.0))
sol = solve(ode, Tsit5())
```

## Getting Started

Explore the example scripts in `examples/` to see the solvers in action:
- `runhdg_poisson.jl` - Poisson equation convergence studies
- `runwavescattering.jl` - Wave scattering on circular domains
- `runeulerchannel.jl` - Compressible Euler equations with shocks
- `runconvection.jl` - Pure convection with DG explicit time-stepping
- `runhdg_ns_kovasznay.jl` - Steady incompressible Navier-Stokes verification (Kovasznay flow, optimal k+1 convergence)
- `runhdg_ns_boussinesq.jl` - Natural convection in a heated cavity (incompressible nonhydrostatic Navier-Stokes with the Boussinesq approximation, validated against the de Vahl Davis benchmark)

Perfect for researchers in numerical analysis, students learning finite element methods, or anyone needing a flexible high-order PDE solver in Julia.