using TwoDG

"""
    square_geometry(m=2, n=2; parity=0) -> MeshGeometry

Geometry of the unit square on an `m × n` vertex grid, with boundaries named
`:bottom`, `:right`, `:top`, `:left` (tags 1–4). `parity` selects the
diagonal direction (see [`make_square_mesh`](@ref)).
"""
function square_geometry(m=2, n=2; parity=0)
    p, t = make_square_mesh(m, n, parity)

    boundary_ϵ = 2e-2
    boundaries = (
        bottom = p -> (p[:, 2]) .< boundary_ϵ,
        right  = p -> (p[:, 1]) .> 1 - boundary_ϵ,
        top    = p -> (p[:, 2]) .> 1 - boundary_ϵ,
        left   = p -> (p[:, 1]) .< boundary_ϵ,
    )

    return MeshGeometry(p, t; boundaries)
end

"""
    mkmesh_square(m=2, n=2, porder=1, parity=0, nodetype=0) -> Mesh

Solver-ready mesh of the unit square: `discretize(square_geometry(m, n; parity), porder; nodetype)`.
"""
mkmesh_square(m=2, n=2, porder=1, parity=0, nodetype=0) =
    discretize(square_geometry(m, n; parity), porder; nodetype)

"""
    mkmesh_distort!(mesh, wig=0.05)

Distort a unit-square mesh (from [`mkmesh_square`](@ref)) in place with a
smooth sinusoidal warp of amplitude `wig`, keeping the boundary fixed.
Useful for testing solver accuracy on non-affine element mappings.
"""
function mkmesh_distort!(mesh, wig=0.05)
    # Computing distortion for mesh vertices
    dx = @. -wig * sin(2π * (mesh.p[:,2] - 0.5)) * cos(π * (mesh.p[:,1] - 0.5))
    dy = @. wig * sin(2π * (mesh.p[:,1] - 0.5)) * cos(π * (mesh.p[:,2] - 0.5))
    @. mesh.p[:,1] += dx
    @. mesh.p[:,2] += dy

    # Computing distortion for cell centers
    dx = @. -wig * sin(2π * (mesh.pcg[:,2] - 0.5)) * cos(π * (mesh.pcg[:,1] - 0.5))
    dy = @. wig * sin(2π * (mesh.pcg[:,1] - 0.5)) * cos(π * (mesh.pcg[:,2] - 0.5))
    @. mesh.pcg[:,1] += dx
    @. mesh.pcg[:,2] += dy

    # Computing distortion for DG nodes
    for i in axes(mesh.dgnodes, 3)
        dx = @. -wig * sin(2π * (mesh.dgnodes[:,2,i] - 0.5)) * cos(π * (mesh.dgnodes[:,1,i] - 0.5))
        dy = @. wig * sin(2π * (mesh.dgnodes[:,1,i] - 0.5)) * cos(π * (mesh.dgnodes[:,2,i] - 0.5))
        @. mesh.dgnodes[:,1,i] += dx
        @. mesh.dgnodes[:,2,i] += dy
    end

    # Mark all faces and elements as curved
    mesh.fcurved .= fill(true, size(mesh.f, 1))
    mesh.tcurved .= fill(true, size(mesh.t, 1))

    return mesh
end
