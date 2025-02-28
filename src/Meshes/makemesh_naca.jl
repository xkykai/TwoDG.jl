using TwoDG
using CSV
using DataFrames
using CairoMakie
using LinearAlgebra
    
function naca0012(x, t=10)
    """Generate y-coordinates for NACA0012 airfoil at given x-coordinates"""
    y = 0.05 * t * (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x^2 + 0.2843 * x^3 - 0.1015 * x^4)
    return y
end

function gmsh_naca(t=10, name="naca0012", display_gmsh=false)
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add(name)

    # Parameters
    chord = 1.0  # Chord length
    n_points = 100  # Reduced number of points on each surface
    mesh_size_airfoil = 0.5  # Increased base mesh size near airfoil
    mesh_size_le = 0.05  # Slightly increased mesh size at leading edge
    mesh_size_te = 0.05  # Slightly increased mesh size at trailing edge
    mesh_size_farfield = 0.2  # Increased mesh size at far-field

    # Generate airfoil points
    x = range(0, chord, length=n_points)
    y_upper = naca0012.(x, t)
    y_lower = -y_upper

    # Add points
    points = Int[]
    # Upper surface (from trailing edge to leading edge)
    for i in reverse(1:n_points)
        point_tag = gmsh.model.geo.addPoint(x[i], y_upper[i], 0, mesh_size_airfoil)
        push!(points, point_tag)
    end

    # Lower surface (from leading edge to trailing edge)
    for i in 2:n_points
        point_tag = gmsh.model.geo.addPoint(x[i], y_lower[i], 0, mesh_size_airfoil)
        push!(points, point_tag)
    end

    # Create airfoil splines
    airfoil_lines = Int[]

    # Upper surface spline (from trailing edge to leading edge)
    upper_line = gmsh.model.geo.addSpline(points[1:n_points])
    push!(airfoil_lines, upper_line)

    # Lower surface spline (from leading edge to trailing edge)
    # We need to make sure this connects properly with the upper surface
    lower_points = vcat([points[n_points]], points[n_points+1:end], [points[1]])
    lower_line = gmsh.model.geo.addSpline(lower_points)
    push!(airfoil_lines, lower_line)

    # Add mesh size fields for refinement
    # Leading edge point field
    le_point_tag = gmsh.model.geo.addPoint(0.0, 0.0, 0, mesh_size_le)
    field_le = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field_le, "PointsList", [le_point_tag])

    # Trailing edge point field
    te_point_tag = gmsh.model.geo.addPoint(1.0, 0.0, 0, mesh_size_te)
    field_te = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field_te, "PointsList", [te_point_tag])

    # Airfoil curve field for overall surface refinement
    field_airfoil = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field_airfoil, "CurvesList", airfoil_lines)
    gmsh.model.mesh.field.setNumber(field_airfoil, "Sampling", 100)  # Number of sampling points

    # Create threshold fields with faster transitions
    field_threshold_le = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(field_threshold_le, "IField", field_le)
    gmsh.model.mesh.field.setNumber(field_threshold_le, "LcMin", mesh_size_le)
    gmsh.model.mesh.field.setNumber(field_threshold_le, "LcMax", mesh_size_airfoil)
    gmsh.model.mesh.field.setNumber(field_threshold_le, "DistMin", 0.2)  # Increased minimum distance
    gmsh.model.mesh.field.setNumber(field_threshold_le, "DistMax", 0.5)  # Increased maximum distance
    gmsh.model.mesh.field.setNumber(field_threshold_le, "StopAtDistMax", 1)  # Ensure smooth transition

    field_threshold_te = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(field_threshold_te, "IField", field_te)
    gmsh.model.mesh.field.setNumber(field_threshold_te, "LcMin", mesh_size_te)
    gmsh.model.mesh.field.setNumber(field_threshold_te, "LcMax", mesh_size_airfoil)
    gmsh.model.mesh.field.setNumber(field_threshold_te, "DistMin", 0.2)  # Increased minimum distance
    gmsh.model.mesh.field.setNumber(field_threshold_te, "DistMax", 0.5)  # Increased maximum distance
    gmsh.model.mesh.field.setNumber(field_threshold_te, "StopAtDistMax", 1)  # Ensure smooth transition

    # Threshold field for entire airfoil surface
    field_threshold_airfoil = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(field_threshold_airfoil, "IField", field_airfoil)
    gmsh.model.mesh.field.setNumber(field_threshold_airfoil, "LcMin", 0.08)  # Mesh size at airfoil surface
    gmsh.model.mesh.field.setNumber(field_threshold_airfoil, "LcMax", mesh_size_airfoil)
    gmsh.model.mesh.field.setNumber(field_threshold_airfoil, "DistMin", 0.2)  # Distance from surface for finest mesh
    gmsh.model.mesh.field.setNumber(field_threshold_airfoil, "DistMax", 1)   # Distance for mesh size transition
    gmsh.model.mesh.field.setNumber(field_threshold_airfoil, "StopAtDistMax", 1)  # Ensure smooth transition

    # Add a gradient field for smoother size transitions
    field_grad = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(field_grad, "F", "0.1 + 0.2*sqrt(x*x + y*y)")

    # Combine the fields using a minimum field
    field_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(field_min, "FieldsList", [field_threshold_le, field_threshold_te, field_threshold_airfoil, field_grad])

    # Set the background field
    gmsh.model.mesh.field.setAsBackgroundMesh(field_min)

    # Additional mesh smoothing options
    gmsh.option.setNumber("Mesh.Smoothing", 100)  # Increase number of smoothing steps
    gmsh.option.setNumber("Mesh.SmoothRatio", 1.3)  # Lower ratio for smoother transition
    gmsh.option.setNumber("Mesh.MeshSizeFactor", 1.0)  # Global mesh size factor

    # Mesh algorithm selection and optimization settings
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D
    gmsh.option.setNumber("Mesh.MeshSizeFactor", 1.0)  # Global mesh size factor
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)  # Don't extend mesh size from boundaries
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)  # Don't compute mesh size from points
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)  # Don't compute mesh size from curvature

    # Create rectangular far-field boundary
    # Define rectangle dimensions
    x_min = -3  # Distance upstream of airfoil
    x_max = 5  # Distance downstream of airfoil
    y_min = -3  # Distance below airfoil
    y_max = 3  # Distance above airfoil

    # Create corner points for rectangle
    far_field_points = Int[]
    # Adding points counter-clockwise from bottom-left
    push!(far_field_points, gmsh.model.geo.addPoint(x_min, y_min, 0, mesh_size_farfield))  # Bottom-left
    push!(far_field_points, gmsh.model.geo.addPoint(x_max, y_min, 0, mesh_size_farfield))  # Bottom-right
    push!(far_field_points, gmsh.model.geo.addPoint(x_max, y_max, 0, mesh_size_farfield))  # Top-right
    push!(far_field_points, gmsh.model.geo.addPoint(x_min, y_max, 0, mesh_size_farfield))  # Top-left

    # Create lines for rectangle edges
    far_field_lines = Int[]
    for i in 1:4
        next_point = i == 4 ? far_field_points[1] : far_field_points[i+1]
        line = gmsh.model.geo.addLine(far_field_points[i], next_point)
        push!(far_field_lines, line)
    end

    # Create curve loop and plane surface
    curve_loop = gmsh.model.geo.addCurveLoop(vcat(airfoil_lines, far_field_lines))
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    # Generate mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)  # 2D mesh

    # Save mesh
    gmsh.write("$(name).msh")

    # Optional: Create physical groups
    airfoil_tag = gmsh.model.addPhysicalGroup(1, airfoil_lines, 1)  # Pass tag as direct argument
    farfield_tag = gmsh.model.addPhysicalGroup(1, far_field_lines, 2)
    domain_tag = gmsh.model.addPhysicalGroup(2, [surface], 3)

    gmsh.model.setPhysicalName(1, airfoil_tag, "Airfoil")
    gmsh.model.setPhysicalName(1, farfield_tag, "FarField")
    gmsh.model.setPhysicalName(2, domain_tag, "FluidDomain")

    # # Optional: Show the mesh in the Gmsh GUI
    if display_gmsh
        gmsh.fltk.run()
    end

    gmsh.finalize()
end

function read_gmsh_mesh(filepath)
    # Initialize Gmsh
    gmsh.initialize()
    
    # Open the mesh file
    gmsh.open(filepath)
    
    # Get nodes
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    
    # Reshape node coordinates from [x1,y1,z1,x2,y2,z2,...] to Array{Float64,2}
    p = reshape(nodeCoords, 3, :)' # Each row is [x,y,z]
    
    # Get 2D elements (triangles)
    elementTypes, elementTags, elementNodeTags = gmsh.model.mesh.getElements(2)  # 2 for 2D elements
    
    # Reshape triangle vertices into an Array{Int,2}
    t = Int.(reshape(elementNodeTags[1], 3, :)')  # Each row is [v1,v2,v3]
    
    # Convert from 1-based to 0-based indexing if needed
    # t = t .- 1
    
    # Finalize Gmsh
    gmsh.finalize()
    
    return p[:, 1:2], t
    # return nodeCoords, elementNodeTags
end

function mkmesh_naca(t_naca = 10, porder=2, name="naca0012", display_gmsh=false)
    gmsh_naca(t_naca, name, display_gmsh)
    p, t = read_gmsh_mesh("$name.msh")
    p, t = fixmesh(p, t)

    f, t2f = mkt2f(t)
    
    function boundary_airfoil(p)
        xs, ys = p[:, 1], p[:, 2]
        outputs = falses(size(p, 1))
        for i in 1:size(p, 1)
            outputs[i] = xs[i] >= 0 && ((ys[i] - naca0012(abs(xs[i]), t_naca)) ^ 2 < 1e-4 || (ys[i] + naca0012(abs(xs[i]))) ^ 2 < 1e-4)
        end
        return outputs
    end
    
    boundary_ϵ = 1e-1
    boundary_left(p) = (p[:, 1]) .< -3 + boundary_ϵ
    boundary_right(p) = (p[:, 1]) .> 5 - boundary_ϵ
    boundary_bottom(p) = (p[:, 2]) .< -3 + boundary_ϵ
    boundary_top(p) = (p[:, 2]) .> 3 - boundary_ϵ
    
    bndexpr = [boundary_airfoil, boundary_left, boundary_right, boundary_bottom, boundary_top]
    
    f = setbndnbrs(p, f, bndexpr)
    
    fcurved = f[:, 4] .== -1
    tcurved = falses(size(t, 1))
    tcurved[f[fcurved, 3]] .= true
    
    plocal, tlocal = uniformlocalpnts(porder)
    
    mesh = TwoDG.Mesh(p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal)

    fd_left(p) = abs(p[1] + 3)
    fd_right(p) = abs(p[1] - 5)
    fd_bottom(p) = abs(p[2] + 3)
    fd_top(p) = abs(p[2] - 3)

    # fd_airfoil(p) = min(abs(naca0012(p[1], t_naca) - p[2]), abs(-naca0012(p[1], t_naca) - p[2]))
    function fd_airfoil(p)
        if p[2] > 0
            return (naca0012(abs(p[1]), t_naca) - p[2])^2
        else
            # @info "Lower"
            # @info -naca0012(p[1], t_naca)
            # @info p[2]
            return (-naca0012(p[1], t_naca) - p[2])^2
        end
    end

    fds = [fd_airfoil, fd_left, fd_right, fd_bottom, fd_top]

    mesh = createnodes(mesh)

    return mesh
end