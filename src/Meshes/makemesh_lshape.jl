using TwoDG

"""
    lshape_geometry(m=2; parity=0) -> MeshGeometry

Geometry of the unit L-shape assembled from three subsquares of `m × m`
vertices each. The whole boundary carries the single name `:boundary`.
"""
function lshape_geometry(m=2; parity=0)
    @assert m >= 2

    p, t = make_square_mesh(m, m, parity)
    num_p = size(p, 1)
    p = p .* 0.5

    p = vcat(
        p,
        hcat(p[:,1], p[:,2] .+ 0.5),
        hcat(p[:,1] .+ 0.5, p[:,2] .+ 0.5)
    )

    t = vcat(t, t .+ num_p, t .+ 2 * num_p)

    p, t = fixmesh(p, t)

    boundaries = (boundary = p -> p[:,2] .< 1.0e6,)

    return MeshGeometry(p, t; boundaries)
end

"""
    mkmesh_lshape(m=2, porder=1, parity=0, nodetype=1) -> Mesh

Solver-ready mesh of the unit L-shape: `discretize(lshape_geometry(m; parity), porder; nodetype)`.
"""
mkmesh_lshape(m=2, porder=1, parity=0, nodetype=1) =
    discretize(lshape_geometry(m; parity), porder; nodetype)
