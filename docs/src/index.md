# TwoDG.jl

*High-order Galerkin methods for PDEs in 2D and 3D — continuous,
discontinuous, and hybridizable discontinuous — behind a single `solve`
entry point, running on CPU or GPU from the same code.*

## Highlights

- **Hybridizable DG (HDG) as a first-class solver.** Implicit high-order
  solves where the globally coupled unknowns live only on element faces:
  static condensation shrinks the system dramatically, the trace system is
  solved directly or with preconditioned Krylov iterations, and a cheap local
  postprocessing step recovers a solution that converges one order *faster*
  than the polynomial degree suggests (``k+2`` superconvergence). This
  extends all the way to steady and unsteady incompressible Navier–Stokes
  with an exactly divergence-free postprocessed velocity.
- **GPU-resident implicit and explicit solvers.** Not just the explicit DG
  time loop: the batched HDG assembly and recovery and the matrix-free CG
  iteration also run through KernelAbstractions, so the same code executes
  on CPU threads or a CUDA GPU (`ArrayT = CuArray`), with no per-backend
  forks.
- **High-order curved simplices, in 2D and 3D.** Triangles and tetrahedra
  with isoparametric curved boundaries at arbitrary polynomial order, so
  p-refinement on circles, airfoils, spheres, and mapped geometries keeps
  its design accuracy. The whole stack is dimension-generic — the dimension
  flows from the mesh, and the same equations, boundary conditions, and
  solvers run on both (see [3D in TwoDG](manual/threed.md)).
- **Three methods, one API.** CG, explicit (L)DG, and implicit HDG share the
  same meshes, equations, and boundary conditions — comparing methods on the
  *same* problem is a few lines of code.
- **An open physics surface.** A new equation, numerical flux, or boundary
  condition defined in *your own script* — using only exported methods —
  runs in every solver path, CPU and GPU alike (see
  [Extending TwoDG](manual/extending.md)).
- **Composable run-time callbacks.** Progress heartbeats, quadrature-exact
  analysis (conservation drift, energies, L² errors), solution snapshots,
  CFL-driven step control, and atomic checkpoint/restart, in the spirit of
  Trixi.jl and Oceananigans.jl (see
  [Callbacks and diagnostics](manual/callbacks.md)).
- **Numerics you can hand a precision or a stepper.** The element type is
  parametric (`T = Float32` runs the whole loop in single precision, on GPU
  too), and [`semidiscretize`](@ref) hands the semidiscrete system to the
  SciML ecosystem when you want adaptive or specialized time integrators.

## Installation

TwoDG is not yet registered; install it directly from GitHub:

```julia
julia> ]  # enter Pkg mode
pkg> add https://github.com/xkykai/TwoDG.jl
```

Julia 1.10 or newer is required. Optional capabilities activate through
package extensions:

| Capability | Activate with |
|---|---|
| Plotting (`scaplot`, `meshplot`) | `using CairoMakie` (any Makie backend) |
| GPU execution | `using CUDA` (or another KernelAbstractions backend) |
| `semidiscretize` → OrdinaryDiffEq | `using SciMLBase` (or any OrdinaryDiffEq solver package) |
| NACA airfoil meshes | `using Gmsh` |

## Gallery

<!-- Images are served from the repository so the docs build stays small.
     These URLs are the only name-dependent strings outside docs/make.jl. -->

```@raw html
<p align="center">
  <img src="https://raw.githubusercontent.com/xkykai/TwoDG.jl/main/figures/eulerchannel_machnumber.gif" height="300" />
  <br>
  <em>Compressible flow through a channel with a bump (2D Euler, Mach number)</em>
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/xkykai/TwoDG.jl/main/figures/hdg_ns_boussinesq_temperature.png" height="320" />
  <img src="https://raw.githubusercontent.com/xkykai/TwoDG.jl/main/figures/hdg_ns_boussinesq_speed.png" height="320" />
  <br>
  <em>Natural convection in a differentially heated cavity at Ra = 10⁴
  (incompressible Navier–Stokes with Boussinesq buoyancy, HDG k = 3):
  temperature (left) and speed (right)</em>
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/xkykai/TwoDG.jl/main/figures/hdg_ns_boussinesq3d_ra1e5.gif" height="300" />
  <br>
  <em>3D natural convection in a differentially heated cavity at Ra = 10⁵
  (incompressible Navier–Stokes with Boussinesq buoyancy, HDG on
  tetrahedra)</em>
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/xkykai/TwoDG.jl/main/figures/hdg_ns_kovasznay_convergence.png" width="800" />
  <br>
  <em>Kovasznay-flow verification of the HDG Navier–Stokes solver: optimal
  k+1 convergence and k+2 superconvergence of the divergence-free
  postprocessed velocity u*</em>
</p>
```

## Supported features

All four solver families are dimension-generic — the same equations and
problems run on triangles and tetrahedra:

| | Equations | Time / solver | Backends |
|---|---|---|---|
| **DG / LDG** (explicit) | convection, convection–diffusion (LDG), first-order wave system, compressible Euler (Roe flux) | internal [`RK4`](@ref), or any OrdinaryDiffEq stepper via [`semidiscretize`](@ref); run-time [callbacks](manual/callbacks.md) | CPU + GPU (KernelAbstractions) |
| **HDG** (implicit, static condensation) | Poisson, steady convection–diffusion | [`Direct`](@ref) sparse LU, [`GMRES`](@ref) (Krylov.jl, block-Jacobi preconditioned, batched assembly) | CPU + GPU trace solve |
| **HDG Navier–Stokes** | steady/unsteady incompressible NS, Boussinesq buoyancy; superconvergent H(div) postprocessing (2D) | Newton + batched direct/GMRES (driver-level API, see `examples/hdg/` and `examples/hdg3d/`) | CPU + GPU local solves (trace LU on CPU) |
| **CG** | Poisson, convection–diffusion–reaction | [`Direct`](@ref) sparse Cholesky/LU, [`ConjugateGradient`](@ref) / [`GMRES`](@ref) (matrix-free) | `Direct()`: CPU; iterative: CPU + GPU |

Meshes: structured square/L-shape and tetrahedral boxes, unstructured circle
(distmesh), cos²-bump duct, Trefftz airfoil (conformal map), NACA 4-digit
and tetrahedral Gmsh import (package extension), uniform red refinement
([`uniref`](@ref)). All support curved isoparametric elements at arbitrary
`porder`; generators attach named boundary tags ([`boundary_names`](@ref)).
A [`MeshGeometry`](@ref) + [`discretize`](@ref) two-stage API separates
geometry from discretization; 3D results export to ParaView as high-order
Lagrange cells ([`save_vtk`](@ref), WriteVTK extension).

## Where to go next

- [Getting started](getting_started.md) — install, first solve, first plot.
- Manual: [Meshes](manual/meshes.md), [Equations and boundary
  conditions](manual/equations.md), [Extending TwoDG](manual/extending.md),
  [Solvers](manual/solvers.md),
  [Callbacks and diagnostics](manual/callbacks.md),
  [3D in TwoDG](manual/threed.md), [GPU support](manual/gpu.md),
  [Plotting](manual/plotting.md).
- [Public API reference](reference/api.md).
- More examples live in the repository's
  [`examples/`](https://github.com/xkykai/TwoDG.jl/tree/main/examples) tree
  (convergence studies, wave scattering, channel flows, Navier–Stokes).
