# Public API

The exported names, grouped by module. The high-level problem/`solve` layer
(first section) is the recommended entry point; the sections after it are the
lower-level building blocks it wraps.

## Problems, equations, and `solve`

```@autodocs
Modules = [TwoDG.Interface]
Private = false
```

## Callbacks and run-time diagnostics

Schedules, the callback catalog, and the quadrature-exact integral
diagnostics of the [Callbacks & diagnostics](../manual/callbacks.md) manual
page.

```@autodocs
Modules = [TwoDG.Callbacks]
Private = false
```

## Meshes

Mesh generators, the two-stage `MeshGeometry`/`discretize` pipeline, and
mesh-connectivity utilities.

```@autodocs
Modules = [TwoDG.Meshes]
Private = false
```

## Master elements

Reference-element shape functions, quadrature rules, and node sets.

```@autodocs
Modules = [TwoDG.Masters]
Private = false
```

## Discontinuous Galerkin

Explicit DG/LDG residuals (legacy and KernelAbstractions paths), the
precomputed [`DGContext`](@ref) geometry, and RK4 time steppers.

```@autodocs
Modules = [TwoDG.DiscontinuousGalerkin]
Private = false
```

## Hybridizable Discontinuous Galerkin

HDG solvers (direct, GMRES, batched), local solves, postprocessing, and the
Navier–Stokes drivers.

```@autodocs
Modules = [TwoDG.HybridizableDiscontinuousGalerkin]
Private = false
```

## Continuous Galerkin

```@autodocs
Modules = [TwoDG.ContinuousGalerkin]
Private = false
```

## Equations, fluxes, and boundary conditions

The dispatch-based physics layer consumed by the DG/HDG kernels: equation
types, numerical fluxes, boundary conditions, and the extension contract for
user-defined physics.

```@autodocs
Modules = [TwoDG.Equations]
Private = false
```

## Plotting

Implemented in the Makie package extension; these stubs error with a load
hint until a Makie backend is loaded.

```@autodocs
Modules = [TwoDG.Plotting]
Private = false
```

## Utilities and drivers

```@autodocs
Modules = [TwoDG.Utils, TwoDG.Drivers]
Private = false
```
