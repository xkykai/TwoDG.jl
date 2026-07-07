# Internals

Non-exported machinery, documented for contributors. Nothing on this page is
part of the public API contract.

## Geometry

The single implementation of face/element geometry (normals, Jacobians,
quadrature-point maps) consumed by `DGContext` and the batched HDG assembly.

```@autodocs
Modules = [TwoDG.Geometry]
```

## Private helpers

```@autodocs
Modules = [TwoDG.Interface, TwoDG.Meshes, TwoDG.Masters,
           TwoDG.DiscontinuousGalerkin,
           TwoDG.HybridizableDiscontinuousGalerkin,
           TwoDG.ContinuousGalerkin, TwoDG.Equations, TwoDG.Callbacks,
           TwoDG.Plotting, TwoDG.Utils, TwoDG.Drivers]
Public = false
```
