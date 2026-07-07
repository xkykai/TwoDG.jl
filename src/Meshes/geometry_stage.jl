# Mesh generation and discretization as separate stages.
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
struct MeshGeometry{Dim, P, T, B, FD}
    p              :: P
    t              :: T
    boundary_names :: Vector{Symbol}
    bndexpr        :: B
    curved         :: Vector{Symbol}
    fd             :: FD

    function MeshGeometry{Dim}(p::P, t::T, boundary_names, bndexpr::B, curved,
                               fd::FD) where {Dim, P, T, B, FD}
        return new{Dim, P, T, B, FD}(p, t, boundary_names, bndexpr, curved, fd)
    end
end

Base.ndims(::MeshGeometry{Dim}) where {Dim} = Dim

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
    return MeshGeometry{size(p, 2)}(p, t, names, bndexpr, curved_names, fd)
end

Base.show(io::IO, geo::MeshGeometry{Dim}) where {Dim} =
    print(io, "MeshGeometry{", Dim, "}(", size(geo.p, 1), " vertices, ", size(geo.t, 1),
          Dim == 2 ? " triangles" : " tetrahedra", ", boundaries = ",
          Tuple(geo.boundary_names), ")")

"""
    discretize(geo::MeshGeometry, porder; nodetype=0) -> Mesh

Discretization stage: build the solver-ready [`Mesh`](@ref) from a
[`MeshGeometry`](@ref) — face/element connectivity (`f`, `t2f`, `t2o`,
`elcon`, `f2f`), high-order nodes (`dgnodes`, projected onto curved
boundaries in 2D), and CG numbering (`pcg`, `tcg`). `nodetype = 0` places
nodes uniformly, `1`/`2` use extended Chebyshev points (2D only).

For a `MeshGeometry{3}` the mesh is tetrahedral: face orientations carry the
6-code `t2o` and the HDG trace connectivity goes through the triangle face
element. Curved boundaries project the boundary-face nodes onto the signed-
distance zero set; nodes on edges shared by *two different* curved
boundaries are projected onto the intersection curve by alternating
projections and written back to every element that shares them (the
edge-before-face rule).
"""
function discretize(geo::MeshGeometry{3}, porder::Integer; nodetype::Integer=0)
    f, t2f, t2o = mkt2f(geo.t)
    f = setbndnbrs(geo.p, f, geo.bndexpr)

    curvedbnd = falses(size(f, 1))          # faces on curved boundaries
    for name in geo.curved
        tag = findfirst(==(name), geo.boundary_names)
        curvedbnd .|= f[:, 5] .== -tag
    end

    plocal = localpnts3d(porder, nodetype)
    onface = ntuple(s -> findall(<(1e-6), @view plocal[:, s]), 4)

    # straight (affine) high-order nodes: the barycentric image of the vertices
    npl = size(plocal, 1)
    nt = size(geo.t, 1)
    dgnodes = zeros(eltype(geo.p), npl, 3, nt)
    for it in 1:nt
        dgnodes[:, :, it] .= plocal * geo.p[geo.t[it, :], :]
    end

    # curved boundaries: project the boundary-face nodes onto the true
    # geometry (isoparametric elements); boundary vertices are expected on
    # the true boundary already, as in 2D. The curvature flags for the metric
    # are derived from which nodes actually moved: unlike in 2D, an element
    # whose only contact with the curved boundary is an *edge* still becomes
    # non-affine (its interior faces acquire curved edges), and treating it as
    # straight breaks the volume/face metric consistency (free stream).
    fcurved = falses(size(f, 1))
    tcurved = falses(nt)
    if any(curvedbnd)
        moved = _project_curved_nodes3d!(dgnodes, geo, f, t2f, curvedbnd, onface)
        for el in 1:nt
            tcurved[el] = any(@view moved[:, el])
        end
        for i in axes(f, 1)
            el = f[i, 4]
            s = findfirst(==(i), @view t2f[el, :])
            fcurved[i] = any(moved[ipl, el] for ipl in onface[s])
        end
    end

    # trace connectivity through the face (triangle) node set: its restriction
    # of the volume lattice, in face-1 barycentric coordinates
    onface = findall(<(1e-6), @view plocal[:, 1])
    fv = face_vertices(Val(3), 1)
    face_plocal = plocal[onface, collect(fv)]
    elcon = mkelcon(t2f, t2o, porder, face_plocal)

    mesh = Mesh(; p=copy(geo.p), t=copy(geo.t), f, t2f, t2o, fcurved, tcurved,
                porder, plocal, tlocal=nothing, dgnodes, elcon,
                f2f=mkf2f(f, t2f), boundary_names=copy(geo.boundary_names))
    return cgmesh(mesh)
end

# Project the high-order nodes of curved boundary faces onto their signed-
# distance zero sets. Nodes are grouped by geometric position, so each point
# is projected exactly once — with *all* of its constraints when it sits on
# several distinct curved boundaries (boundary-intersection edges land on the
# true curve) — and the result is written back to **every** element copy of
# that point: elements that touch the curved boundary only through an edge or
# a vertex hold copies too, and the mesh stays conforming only if they all
# move together. Returns `moved (npl, nt)`: which node copies changed.
function _project_curved_nodes3d!(dgnodes, geo, f, t2f, curvedbnd, onface)
    fd = collect(geo.fd)

    # position quantization: far coarser than roundoff scatter between copies
    # of one node (~1e-15), far finer than any node separation
    scale = max(maximum(abs, geo.p), one(eltype(geo.p)))
    key(x) = ntuple(d -> round(Int, x[d] / scale * 1e8), 3)

    # (position key) -> (affine start position, curved tags meeting there)
    groups = Dict{NTuple{3, Int}, Tuple{Vector{Float64}, Vector{Int}}}()
    for i in findall(curvedbnd)
        el = f[i, 4]
        tag = -f[i, 5]
        s = findfirst(==(i), @view t2f[el, :])
        for ipl in onface[s]
            x = dgnodes[ipl, :, el]
            _, tags = get!(() -> (x, Int[]), groups, key(x))
            tag in tags || push!(tags, tag)
        end
    end

    # project each distinct point once, with all of its constraints
    proj = Dict{NTuple{3, Int}, Vector{Float64}}()
    for (k, (x0, tags)) in groups
        x = copy(x0)
        if length(tags) == 1
            x = project_to_boundary(fd[tags[1]], x, 0)
        else
            # intersection of several curved boundaries: alternating projection
            for _ in 1:100
                xp = copy(x)
                for tag in tags
                    x = project_to_boundary(fd[tag], x, 0)
                end
                norm(x - xp) < 1e-13 * scale && break
            end
        end
        proj[k] = x
    end

    # global write-back; probing the ±1 neighbor keys guards against copies
    # of one node straddling a quantization boundary
    npl, _, nt = size(dgnodes)
    moved = falses(npl, nt)
    offs = ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for el in 1:nt, ipl in 1:npl
        k = key(@view dgnodes[ipl, :, el])
        for o in offs
            x = get(proj, (k[1] + o[1], k[2] + o[2], k[3] + o[3]), nothing)
            if x !== nothing
                moved[ipl, el] = abs(x[1] - dgnodes[ipl, 1, el]) +
                                 abs(x[2] - dgnodes[ipl, 2, el]) +
                                 abs(x[3] - dgnodes[ipl, 3, el]) > 1e-13 * scale
                dgnodes[ipl, :, el] .= x
                break
            end
        end
    end
    return moved
end

function discretize(geo::MeshGeometry, porder::Integer; nodetype::Integer=0)
    f, t2f, t2o = mkt2f(geo.t)
    f = setbndnbrs(geo.p, f, geo.bndexpr)

    fcurved = falses(size(f, 1))
    for name in geo.curved
        tag = findfirst(==(name), geo.boundary_names)
        fcurved .|= f[:, 4] .== -tag
    end
    tcurved = falses(size(geo.t, 1))
    tcurved[f[fcurved, 3]] .= true

    plocal, tlocal = localpnts(porder, nodetype)
    mesh = Mesh(; p=copy(geo.p), t=copy(geo.t), f, t2f, t2o, fcurved, tcurved,
                porder, plocal, tlocal, boundary_names=copy(geo.boundary_names))
    mesh = createnodes(mesh, geo.fd === nothing ? nothing : collect(geo.fd))
    return cgmesh(mesh)
end
