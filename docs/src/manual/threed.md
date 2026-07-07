# 3D in TwoDG

Everything documented for 2D — the equations, boundary conditions,
`Problem`/`solve` API, GPU support, and the HDG superconvergence machinery —
is dimension-generic and works on tetrahedral meshes. A user never writes the
dimension twice: it flows from the mesh (`Mesh{3}`), and scalar equations
infer it from their data (`ConvectionEquation(v)` with a 3-vector `v` is
three-dimensional). This page collects what is genuinely different in 3D:
where meshes come from, what 3D costs, and how to look at the results.

## Mesh sources

**Structured boxes** — [`mkmesh_box`](@ref)`(nx, ny, nz, porder)` (or
[`box_geometry`](@ref) + [`discretize`](@ref)) tiles a structured grid into
6 tetrahedra per cube (the Kuhn/Freudenthal split: conforming without parity
tricks, uniformly refinable). Boundary names:
`:left/:right/:bottom/:top/:front/:back`.

**Uniform refinement** — [`uniref`](@ref) red-refines each tet into 8
(Bey's rule, splitting the interior octahedron along its shortest diagonal
for shape regularity).

**Gmsh import** — with Gmsh.jl loaded, `gmsh_geometry(path; boundaries,
curved, fd)` reads linear tetrahedral (and triangular) meshes into a
[`MeshGeometry`](@ref), which `discretize` then equips with high-order nodes.

**Curved boundaries** — declare boundaries `curved` in the `MeshGeometry`
and supply signed-distance functions; `discretize` projects the high-order
boundary nodes onto the exact surface (edges shared by two curved patches are
projected onto both, alternating — the edge-before-face rule). Two rules for
*convergence studies* on curved domains:

- Build the refinement family by **smoothly blending** the reference domain
  onto the curved boundary (e.g. the sphere-octant map
  ``v \mapsto v\,(1 - s + s^2/\lVert v\rVert)``, ``s = \sum_i v_i``).
  Projecting boundary vertices *after* refinement stretches the
  boundary-layer tets like ``1/h`` and stalls every method at first order.
- For HDG postprocessing, give the ``p+1`` mesh the ``p``-mesh's
  isoparametric map with [`match_geometry!`](@ref) (see the
  [HDG superconvergence in 3D](../tutorials/threed_hdg.md) tutorial).

## What 3D costs

The polynomial spaces grow from ``(p+1)(p+2)/2`` to ``(p+1)(p+2)(p+3)/6``
nodes per element, and default quadrature (`pgauss = 4p`) from ``O(p^2)`` to
``O(p^3)`` points. At ``p = 3`` on tets: 20 nodes and 343 quadrature points
per element.

Two design decisions keep this affordable:

**Symmetric quadrature.** [`gaussquad3d`](@ref) uses tabulated symmetric
rules (Witherden & Vincent 2015) up to degree 10 — 2–3× fewer points than
collapsed-coordinate product rules of the same degree (46 vs 125 at degree
8, the ``p = 2`` default) — and falls back to product rules above that.
On straight-element meshes, `ReferenceElement(mesh, 2 * mesh.porder + 1)`
is sufficient for the bilinear forms and cuts the point count further; the
generous `4p` default matters mainly for curved elements and strongly
nonlinear fluxes.

**The affine fast path.** Dense per-element derivative tables
(`shapd`, ``npl \times ng \times 3``) cost ~165 KB *per element* at
``p = 3`` — 1 GB for a modest 6000-tet mesh, 16 GB at 100k tets. Since most
elements of any real mesh are straight, [`GeometricFactors`](@ref) stores
dense tables **only for curved elements**; affine elements store their
constant Jacobian data (``O(3^2)`` per element) and share one set of
reference tables. The kernels rotate affine-element fluxes into the
reference frame at quadrature points, so the hot lift loops do the same
FLOPs as before against cache-resident shared tables. On the all-straight
6000-tet ``p = 3`` box this shrinks the geometry cache from **1057 MB to
54 MB** (the remainder is dominated by the per-element inverse mass
matrices, which stay dense by design) with residual timings unchanged. The
pre-split dense arrays remain available as materializing properties
(`ctx.shapd`, `ctx.wjac`, `ctx.pg`) for diagnostics.

Measured on one machine (8-thread laptop CPU, RTX 3050 Ti 4 GB): Euler
residual on a 6000-tet box, `Float32`, ms per evaluation —

| p | `pgauss` (ng) | CPU (8 threads) | GPU | Speedup |
|---|---|---|---|---|
| 2 | 8 (46, default) | 12.8 | 2.7 | 4.8× |
| 2 | 5 (14) | 6.7 | 1.2 | 5.5× |
| 3 | 12 (343, default) | 144 | 17.4 | 8.3× |
| 3 | 7 (35) | 16.8 | 2.4 | 6.9× |
| 4 | 16 (729, default)¹ | 114 | 18.0 | 6.3× |
| 4 | 9 (59)¹ | 8.2 | 1.3 | 6.5× |

¹ 1296 tets. The `pgauss = 2p+1` rows use the symmetric rules end-to-end;
on straight meshes they lose no order of accuracy for the bilinear forms
(the generous default guards curved elements and strongly nonlinear
fluxes — for Euler, verify against a `4p` reference on your problem).
A per-kernel breakdown shows the volume flux + lift at 74–87% of the
residual and the atomic face-scatter at 2–3% — quadrature economy, not
scatter strategy, is the lever that matters.

HDG in 3D condenses onto triangular-face traces (``(p+1)(p+2)/2`` nodes per
face, blocks of ``4 n_{ps} \times 4 n_{ps}``); the batched assembly and
recovery run on CPU or GPU, while the sparse trace factorization stays on
the CPU. Small meshes are launch-overhead-bound on GPUs — judge speedups at
``\gtrsim 5000`` elements. Use `Float32` for explicit DG on consumer cards,
but keep `Float64` for HDG trace solves (memory-bound GMRES; single
precision can stagnate before tight tolerances).

## ParaView output

3D solutions go to ParaView through the WriteVTK extension: with WriteVTK.jl
loaded, [`save_vtk`](@ref) writes high-order Lagrange VTK cells, which
ParaView renders exactly — including curved isoparametric elements and the
polynomial solution within each element (increase *Nonlinear Subdivision
Level* in the display properties to see the curvature).

```julia
using WriteVTK
save_vtk(mesh, sol.u, "flow"; names = (:ρ, :ρu, :ρv, :ρw, :ρE))
save_vtk(sol, "flow")                        # same, names inferred
```

Time series (e.g. from a callback in the time loop) can be collected into a
ParaView `.pvd` collection; see `examples/dg3d/run3d_euler_pulse.jl`.

## What is 2D-only today

- Makie plotting (`meshplot`/`scaplot`) — use ParaView in 3D.
- The H(div) Navier-Stokes velocity postprocessing (`hdg_ns_postprocess`)
  — the scalar HDG postprocessing ([`hdg_postprocess`](@ref)) is fully
  3D.
- `cg_bounds`/`grad_u` utilities.
