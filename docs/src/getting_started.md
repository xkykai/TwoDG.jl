# Getting started

## Install

```julia
julia> ]  # enter Pkg mode
pkg> add https://github.com/xkykai/TwoDG.jl
```

## A first solve: Poisson on the unit square

Every solver in TwoDG follows the same pattern: build a **mesh**, pick an
**equation**, wrap both in a **problem**, and call **`solve`**.

```julia
using TwoDG

# unit square, 8×8×2 triangles of polynomial order 3
mesh = mkmesh_square(9, 9, 3, 0, 1)

# -Δu = f with u = 0 on the boundary, continuous Galerkin
f(x, y) = 2π^2 * sin(π * x) * sin(π * y)
prob = CGProblem(PoissonEquation(), mesh; source = f)
sol  = solve(prob)

l2error(sol, (x, y) -> sin(π * x) * sin(π * y))   # ≈ 3e-7
```

The same problem solved with HDG — an implicit method whose globally coupled
unknowns live only on element faces — takes a quadrature-point source
`p -> values` and a [`Dirichlet`](@ref) boundary object instead:

```julia
fq(p) = 2π^2 .* sin.(π .* p[:, 1]) .* sin.(π .* p[:, 2])
prob = HDGProblem(PoissonEquation(), mesh; bc = Dirichlet(0.0), source = fq)
sol  = solve(prob)              # GMRES by default; solve(prob, Direct()) for sparse LU
```

## A time-dependent problem

Time-dependent conservation laws use [`DGProblem`](@ref) with boundary
conditions **by name** (the square mesh tags its sides `:bottom`, `:right`,
`:top`, `:left`) and a CFL-based time step:

```julia
eq   = ConvectionDiffusionEquation([1.0, 0.5], 0.01)
prob = DGProblem(eq, mesh;
                 bc = (bottom = Dirichlet(), right = Neumann(),
                       top    = Dirichlet(), left  = Neumann()),
                 u0 = [(x, y) -> exp(-16 * ((x - 0.5)^2 + (y - 0.5)^2))])
sol  = solve(prob, RK4(); dt = compute_dt(prob), tfinal = 1.0)
```

Prefer an adaptive stepper? Hand the semidiscretization to OrdinaryDiffEq
(this needs SciMLBase loaded, which any OrdinaryDiffEq package provides):

```julia
using OrdinaryDiffEqTsit5
ode = semidiscretize(prob, (0.0, 1.0))
sol = solve(ode, Tsit5())
```

## Plotting

Plotting activates when a Makie backend is loaded:

```julia
using CairoMakie
scaplot(mesh, sol.u[:, 1, :], show_mesh = true)
```

## Running on a GPU

Pass a device array type to `solve` and the whole time loop runs on the
device:

```julia
using CUDA
sol = solve(prob, RK4(); dt = compute_dt(prob), tfinal = 1.0, ArrayT = CuArray)
```

Single precision end to end: construct the problem with `T = Float32`. See
the [GPU manual page](manual/gpu.md) for details on what moves to the device.

## Where next

- [Meshes](manual/meshes.md) — generators, named boundaries, curved
  elements, the `MeshGeometry`/`discretize` two-stage API.
- [Equations and boundary conditions](manual/equations.md) — the built-in
  equations, fluxes, and boundary conditions.
- [Extending TwoDG](manual/extending.md) — define your own equation,
  numerical flux, or boundary condition in your own script.
- [Solvers](manual/solvers.md) — DG/LDG, HDG (with superconvergent
  postprocessing), CG, and the Navier–Stokes drivers.
- [Callbacks and diagnostics](manual/callbacks.md) — progress, analysis,
  snapshots, and checkpoint/restart while a solve runs.
- [3D in TwoDG](manual/threed.md) — everything above on tetrahedra.
- The repository's `examples/` tree for complete scripts.
