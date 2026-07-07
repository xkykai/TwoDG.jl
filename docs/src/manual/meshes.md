# Meshes

TwoDG works on unstructured simplex meshes — triangles in 2D, tetrahedra in
3D (`Mesh{3}`; this page uses the 2D vocabulary, and
[3D in TwoDG](threed.md) covers what differs) — with high-order, possibly
curved (isoparametric) elements. A solver-ready [`Mesh`](@ref) carries three
layers of data:

1. **Vertex geometry** — vertex coordinates `p (np, Dim)` and simplices
   `t (nt, Dim+1)` (counterclockwise in 2D).
2. **Connectivity** — the face list `f`, element-to-face map `t2f`, HDG trace
   connectivity `elcon`, face-to-face map `f2f`, and the continuous (CG)
   numbering `pcg`/`tcg`.
3. **High-order nodes** — `dgnodes (npl, Dim, nt)`, the coordinates of each
   element's `npl` nodes (`(p+1)(p+2)/2` on triangles), projected onto the
   true boundary for curved elements.

The field-by-field conventions (what the columns of `f` mean, the sign of
`t2f`, boundary tags) are documented on the [`Mesh`](@ref) docstring.

## Built-in generators

| Generator | Domain | Boundary names |
|---|---|---|
| [`mkmesh_square`](@ref) | unit square, structured | `:bottom`, `:right`, `:top`, `:left` |
| [`mkmesh_lshape`](@ref) | unit L-shape | `:boundary` |
| [`mkmesh_circle`](@ref) | unit circle, unstructured (distmesh via Python), curved | `:boundary` |
| [`mkmesh_trefftz`](@ref) | Karman–Trefftz airfoil O-mesh (conformal map), curved | `:airfoil`, `:farfield` |
| [`mkmesh_duct`](@ref) | cos²-bump channel (maps a square mesh), curved | inherited from the square |
| [`mkmesh_naca`](@ref) | NACA 4-digit airfoil via Gmsh (package extension), curved | `:airfoil`, `:left`, `:right`, `:bottom`, `:top` |
| [`mkmesh_box`](@ref) | unit box, structured tetrahedra (3D) | `:left`, `:right`, `:bottom`, `:top`, `:front`, `:back` |

All generators take a polynomial order `porder` and (most) a `nodetype`
(`0` = uniform nodes, `1`/`2` = extended Chebyshev; see
[`localpnts`](@ref)). [`mkmesh_distort!`](@ref) warps a square mesh smoothly
to test non-affine element mappings.

## Named boundaries

Generators attach boundary names, in boundary-tag order, retrievable with
[`boundary_names`](@ref):

```julia
julia> mesh = mkmesh_square(9, 9, 3);

julia> boundary_names(mesh)
4-element Vector{Symbol}:
 :bottom
 :right
 :top
 :left
```

Boundary tag `k` is stored as `-k` in `mesh.f[:, 4]`. The high-level
interface accepts boundary conditions keyed by these names (see
[Equations and boundary conditions](equations.md)).

## The two-stage API: geometry, then discretization

Mesh *generation* and *discretization* are separate stages. A
[`MeshGeometry`](@ref) holds only what a generator knows — vertices,
triangles, named boundary classifiers, and (for curved domains) signed
distance functions — and [`discretize`](@ref) turns it into a solver-ready
`Mesh` at a chosen polynomial order:

```julia
geo  = square_geometry(17, 17)          # geometry stage: no porder yet
mesh3 = discretize(geo, 3)              # p = 3 discretization
mesh5 = discretize(geo, 5; nodetype=1)  # p = 5, Chebyshev nodes, same geometry
```

The one-shot generators are thin wrappers, e.g. `mkmesh_square(m, n, porder,
parity, nodetype)` is exactly `discretize(square_geometry(m, n; parity),
porder; nodetype)`. Building your own geometry:

```julia
geo = MeshGeometry(p, t;
                   boundaries = (wall  = p -> p[:, 2] .< 1e-3,
                                 outer = p -> p[:, 2] .>= 1e-3),
                   curved = [:outer],
                   fd = [p -> 0.0, p -> sqrt(sum(p.^2)) - 2])
mesh = discretize(geo, 4)
```

`boundaries` maps names to classifiers evaluated on face midpoints (their
order defines the tags); `curved` names the boundaries whose high-order
nodes should be projected onto the true boundary; `fd` supplies one signed
distance function per boundary when any boundary is curved.

## Curved elements

Faces flagged in `mesh.fcurved` (and their elements in `mesh.tcurved`) are
treated isoparametrically: during [`discretize`](@ref)/[`createnodes`](@ref)
their `dgnodes` are projected onto the zero level set of the boundary's
distance function, and the solvers evaluate per-quadrature-point Jacobians
on them. Straight elements use the cheaper affine path automatically. This
is what preserves the design order of accuracy on circles and airfoils.

## Gmsh meshes

`mkmesh_naca` lives in a package extension: it becomes available when
Gmsh.jl is loaded (`using TwoDG, Gmsh`). It generates the airfoil geometry,
meshes it with Gmsh, classifies the five boundaries, and projects the
airfoil-adjacent high-order nodes onto the exact NACA surface. The same
read-classify-project pattern is the template for importing other Gmsh
meshes.

## Low-level utilities

[`mkt2f`](@ref) builds the face list and element-to-face map from `t`;
[`setbndnbrs`](@ref) assigns boundary tags from classifier functions;
[`createnodes`](@ref) places (and projects) the high-order nodes;
[`cgmesh`](@ref) derives the deduplicated CG numbering; [`mkf2f`](@ref)
builds face-to-face connectivity for the HDG block-Jacobi preconditioner;
[`uniref`](@ref) uniformly refines a `(p, t)` triangulation; and
[`fixmesh`](@ref) deduplicates vertices and fixes triangle orientation.
