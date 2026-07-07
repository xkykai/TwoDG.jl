# Plotting

Plotting lives in a package extension so the core package never compiles the
Makie stack. Load any Makie backend to activate it:

```julia
using TwoDG
using CairoMakie   # or GLMakie

mesh = mkmesh_square(17, 17, 3, 0, 1)
sol  = solve(CGProblem(PoissonEquation(), mesh;
                       source = (x, y) -> 2π^2 * sin(π * x) * sin(π * y)))

scaplot(mesh, sol.u[:, 1, :]; show_mesh = true)
```

Without a backend loaded, the functions error with a load hint.

- `scaplot(mesh, c; limits=nothing, show_mesh=false, figure_size=(800, 800), title="", cmap=:turbo)`
  — filled contour plot of a scalar field `c (npl, nt)` in DG numbering
  (curved elements rendered curved). Returns the Makie figure, so it
  composes with `save("plot.png", fig)` and animation loops.
- `meshplot(mesh; nodes=false, annotate="")` — wireframe of the
  triangulation; `nodes=true` marks the DG nodes, and `annotate` may contain
  `'p'` (number the vertices) and/or `'t'` (number the elements).
- `meshplot_curved(mesh; nodes=false, annotate="", pplot=0)` — wireframe
  following the curved (isoparametric) element edges, subdividing each
  element with order `pplot` for display; useful for checking boundary
  projection.

All solution objects store their mesh, so the pattern is always
`scaplot(sol.prob.mesh, sol.u[:, component, :])`. For time series, call
`scaplot` inside a loop over saved states (see
`examples/dg/runeulerchannel_animation.jl` for a `record`-based movie).

The Makie plots are 2D-only. 3D solutions go to ParaView via
[`save_vtk`](@ref) (WriteVTK package extension) — see
[3D in TwoDG](threed.md).
