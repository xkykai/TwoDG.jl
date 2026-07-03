using TwoDG

"""
    circle_geometry(p, t) -> MeshGeometry

Geometry of the unit circle from a raw triangulation `(p, t)` (deduplicated
with [`fixmesh`](@ref)). The single boundary is named `:boundary` and is
curved: high-order nodes are projected onto the unit circle during
[`discretize`](@ref).
"""
function circle_geometry(p, t)
    p, t = fixmesh(p, t)

    boundaries = (boundary = p -> sqrt.(sum(p.^2, dims=2)) .> 1 - 2e-2,)
    fd_circle(p) = sqrt(sum(p.^2)) - 1

    return MeshGeometry(p, t; boundaries, curved=[:boundary], fd=[fd_circle])
end

"""
    make_circle_nodes(p, t, porder, nodetype) -> Mesh

Discretize a raw unit-circle triangulation `(p, t)` (e.g. from
[`make_circle_mesh`](@ref)) at order `porder`, projecting the high-order
boundary nodes onto the unit circle. Equivalent to
`discretize(circle_geometry(p, t), porder; nodetype)`.
"""
make_circle_nodes(p, t, porder, nodetype) =
    discretize(circle_geometry(p, t), porder; nodetype)

"""
    mkmesh_circle(siz=0.4, porder=3, nodetype=0; boundary_refinement=nothing) -> Mesh

Unstructured curved mesh of the unit circle (element size `siz`), generated
with distmesh via Python (see `make_circle_mesh`) and discretized at order
`porder`.
"""
function mkmesh_circle(siz=0.4, porder=3, nodetype=0; boundary_refinement=nothing)
    p, t = make_circle_mesh(siz, boundary_refinement)
    return make_circle_nodes(p, t, porder, nodetype)
end
