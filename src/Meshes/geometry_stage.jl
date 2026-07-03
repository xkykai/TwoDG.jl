# Roadmap A2.5: mesh generation and discretization as separate stages.
#
# A `MeshGeometry` carries only what a mesh *generator* knows: vertices,
# triangles, named boundary classifiers, and (for curved domains) signed
# distance functions. `discretize(geo, porder)` turns it into the fully
# populated `Mesh` the solvers consume (connectivity, high-order nodes, CG
# numbering). The legacy `mkmesh_*` entry points are thin wrappers around
# these two stages.

using TwoDG

"""
    MeshGeometry(p, t; boundaries, curved=Symbol[], fd=nothing)

Geometry-stage description of a triangulated domain, before any choice of
polynomial order or node placement.

- `p (np, 2)`: vertex coordinates; `t (nt, 3)`: triangle vertex indices.
- `boundaries`: a `NamedTuple` (or pairs) of `name => classifier`, where
  `classifier(midpoints) -> Bool` selects the boundary faces belonging to
  `name`. The order defines the boundary tags (`-1, -2, …` in `mesh.f`).
- `curved`: names of boundaries that are curved (their high-order face nodes
  are projected onto the true boundary during [`discretize`](@ref)).
- `fd`: `nothing`, or one signed-distance function per boundary (same order
  as `boundaries`); required when `curved` is nonempty.

See [`discretize`](@ref).
"""
struct MeshGeometry{P, T, B, FD}
    p              :: P
    t              :: T
    boundary_names :: Vector{Symbol}
    bndexpr        :: B
    curved         :: Vector{Symbol}
    fd             :: FD
end

function MeshGeometry(p, t; boundaries, curved=Symbol[], fd=nothing)
    names = collect(Symbol, keys(boundaries))
    bndexpr = collect(values(boundaries))
    curved_names = collect(Symbol, curved)
    for name in curved_names
        name in names ||
            throw(ArgumentError("curved boundary :$name is not one of the boundaries $(Tuple(names))"))
    end
    if !isempty(curved_names)
        fd === nothing &&
            throw(ArgumentError("curved boundaries require distance functions `fd` (one per boundary)"))
        length(fd) == length(names) ||
            throw(ArgumentError("`fd` must have one entry per boundary ($(length(names))), got $(length(fd))"))
    end
    return MeshGeometry(p, t, names, bndexpr, curved_names, fd)
end

Base.show(io::IO, geo::MeshGeometry) =
    print(io, "MeshGeometry(", size(geo.p, 1), " vertices, ", size(geo.t, 1),
          " triangles, boundaries = ", Tuple(geo.boundary_names), ")")

"""
    discretize(geo::MeshGeometry, porder; nodetype=0) -> Mesh

Discretization stage: build the solver-ready [`Mesh`](@ref) from a
[`MeshGeometry`](@ref) — face/element connectivity (`f`, `t2f`, `elcon`,
`f2f`), high-order nodes (`dgnodes`, projected onto curved boundaries), and
CG numbering (`pcg`, `tcg`). `nodetype = 0` places nodes uniformly, `1`/`2`
use extended Chebyshev points.
"""
function discretize(geo::MeshGeometry, porder::Integer; nodetype::Integer=0)
    f, t2f = mkt2f(geo.t)
    f = setbndnbrs(geo.p, f, geo.bndexpr)

    fcurved = falses(size(f, 1))
    for name in geo.curved
        tag = findfirst(==(name), geo.boundary_names)
        fcurved .|= f[:, 4] .== -tag
    end
    tcurved = falses(size(geo.t, 1))
    tcurved[f[fcurved, 3]] .= true

    plocal, tlocal = localpnts(porder, nodetype)
    mesh = Mesh(; p=copy(geo.p), t=copy(geo.t), f, t2f, fcurved, tcurved,
                porder, plocal, tlocal, boundary_names=copy(geo.boundary_names))
    mesh = createnodes(mesh, geo.fd === nothing ? nothing : collect(geo.fd))
    return cgmesh(mesh)
end
