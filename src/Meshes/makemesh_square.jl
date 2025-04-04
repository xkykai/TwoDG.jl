using TwoDG

function mkmesh_square(m=2, n=2, porder=1, parity=0, nodetype=0)
    p, t = make_square_mesh(m, n, parity)
    
    f, t2f = mkt2f(t)
    
    boundary_ϵ = 2e-2
    boundary_left(p) = (p[:, 1]) .< boundary_ϵ
    boundary_right(p) = (p[:, 1]) .> 1 - boundary_ϵ
    boundary_bottom(p) = (p[:, 2]) .< boundary_ϵ
    boundary_top(p) = (p[:, 2]) .> 1 - boundary_ϵ
    
    bndexpr = [boundary_left, boundary_right, boundary_bottom, boundary_top]
    
    f = setbndnbrs(p, f, bndexpr)
    
    fcurved = falses(size(f, 1))
    tcurved = falses(size(t, 1))
    tcurved[f[fcurved, 3]] .= true
    
    plocal, tlocal = localpnts(porder, nodetype)
    
    mesh = TwoDG.Mesh(; p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal)
    
    mesh = createnodes(mesh)
    mesh = cgmesh(mesh)

    return mesh
end

function mkmesh_distort!(mesh, wig=0.05)
    """
    mkmesh_distort distorts a unit square mesh keeping boundaries unchanged
    
    # Arguments
    - `mesh`: mesh data structure
        * input: mesh for the unit square created with mkmesh_square
        * output: distorted mesh
    - `wig`: amount of distortion (default: 0.05)
    
    # Returns
    - Distorted mesh
    """
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