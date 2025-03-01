using ForwardDiff
using LinearAlgebra
using TwoDG.Utils: newton_raphson

struct Mesh{P, T, F, TF, FC, TC, PO, PL, TL, DG}
    p::P
    t::T
    f::F
    t2f::TF
    fcurved::FC
    tcurved::TC
    porder::PO
    plocal::PL
    tlocal::TL
    dgnodes::DG
    
    Mesh(p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal) = new{typeof(p), typeof(t), typeof(f), typeof(t2f), 
                                                                       typeof(fcurved), typeof(tcurved), typeof(porder),
                                                                       typeof(plocal), typeof(tlocal), Nothing}(
                                                                       p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal, nothing)

    Mesh(p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal, dgnodes) = new{typeof(p), typeof(t), typeof(f), typeof(t2f), 
                                                                                typeof(fcurved), typeof(tcurved), typeof(porder),
                                                                                typeof(plocal), typeof(tlocal), typeof(dgnodes)}(
                                                                                p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal, dgnodes)

    Mesh(p, t, porder, plocal, tlocal) = new{typeof(p), typeof(t), Nothing, Nothing, Nothing, Nothing, typeof(porder),
                                             typeof(plocal), typeof(tlocal), Nothing}(p, t, nothing, nothing, nothing, nothing, porder, plocal, tlocal, nothing)
    
    Mesh(mesh::Mesh, dgnodes) = new{typeof(mesh.p), typeof(mesh.t), typeof(mesh.f), typeof(mesh.t2f), 
                                    typeof(mesh.fcurved), typeof(mesh.tcurved), typeof(mesh.porder),
                                    typeof(mesh.plocal), typeof(mesh.tlocal), typeof(dgnodes)}(
                                    mesh.p, mesh.t, mesh.f, mesh.t2f, mesh.fcurved, mesh.tcurved, mesh.porder, mesh.plocal, mesh.tlocal, dgnodes)
end

# Converts barycentric coordinates (λ) to Cartesian coordinates using vertices v₁, v₂, v₃
# λ contains the coordinates [λ₂, λ₃], with λ₁ = 1-λ₂-λ₃ implied
function barycentric_to_cartesian(λ, v₁, v₂, v₃)
    # Create transformation matrix from barycentric to Cartesian
    # T maps the barycentric space to the triangle in Cartesian space
    T = hcat(v₂ .- v₁, v₃ .- v₁)
    # Apply transformation and shift by v₁
    return T * λ .+ v₁
end

# Creates a function that returns the derivative of f using forward automatic differentiation
autodiff(f) = x -> ForwardDiff.derivative(f, x)

# Projects a point x₀ onto the boundary defined by distance_function
# s is an initial guess for the distance to the boundary
function project_to_boundary(distance_function, x₀, s=0)
    # Calculate gradient of the distance function at x₀
    grad = ForwardDiff.gradient(distance_function, x₀)
    
    # If gradient is zero, x₀ is already at a critical point
    if grad[1] == 0 && grad[2] == 0
        return x₀
    else
        # Normalize gradient to get direction vector
        grad_norm = grad / norm(grad)
        
        # Define a 1D function along the gradient direction
        fd_linedirection(s) = distance_function(x₀ .+ s .* grad_norm)
        
        # Use Newton-Raphson to find the value of s where fd_linedirection(s) = 0
        # (i.e., find where the point lands on the boundary)
        s = newton_raphson(fd_linedirection, autodiff(fd_linedirection), s)
        
        # Return the projected point
        return x₀ .+ s .* grad_norm
    end
end

# Checks if a barycentric coordinate λ represents a vertex
# λ = [λ₁, λ₂, λ₃] where λ₁ = 1-λ₂-λ₃ is implied
function isvertex(λ)
    λ₁ = λ[2]
    λ₂ = λ[3]
    # A vertex occurs when coordinates are binary (0 or 1)
    return (λ₁ == 0 || λ₁ == 1) && (λ₂ == 0 || λ₂ == 1)
end

# Checks if a barycentric coordinate λ lies on an edge
function isedge(λ)
    # An edge occurs when any barycentric coordinate is zero
    return any(λ .== 0)
end

# Determines which edge the barycentric coordinate λ lies on
function edge_number(λ)
    if λ[3] == 0
        return 1  # Edge between vertices 1 and 2
    elseif λ[2] == 0
        return 3  # Edge between vertices 3 and 1
    else
        return 2  # Edge between vertices 2 and 3
    end
end

# Checks if a point (in barycentric coordinates λ) lies on a curved edge
# vn₁, vn₂, vn₃ are vertex indices, it is the triangle index
function iscurvededge(λ, mesh, vn₁, vn₂, vn₃, it)
    # Get all curved faces from the mesh
    # mesh.fcurved contains indices of curved faces
    all_curved_faces = mesh.f[mesh.fcurved, :]
    
    # Find the row where the third column equals it
    row_number = findfirst(x -> x == it, all_curved_faces[:, 3])
    curved_face = all_curved_faces[row_number, :]
    
    # Determine which edge we're on based on barycentric coordinates
    Eₙ = edge_number(λ)
    
    # Check if the vertices of this edge match the curved face definition
    if Eₙ == 1
        return (vn₁ == curved_face[1] && vn₂ == curved_face[2]) || (vn₁ == curved_face[2] && vn₂ == curved_face[1])
    elseif Eₙ == 2
        return (vn₂ == curved_face[1] && vn₃ == curved_face[2]) || (vn₂ == curved_face[2] && vn₃ == curved_face[1])
    else
        return (vn₃ == curved_face[1] && vn₁ == curved_face[2]) || (vn₃ == curved_face[2] && vn₁ == curved_face[1])
    end
end

# Checks if a point lies on a curved boundary
# In mesh.f, negative values in the 4th column indicate boundaries
function iscurvedboundary(λ, mesh, vn₁, vn₂, vn₃, it)
    Eₙ = edge_number(λ)
    
    # Find where boundary definitions start in mesh.f (indicated by negative values in 4th column)
    i_bnd = findfirst(x -> x < 0, mesh.f[:, 4])
    
    # Get all curved boundaries
    all_curved_boundaries = mesh.f[i_bnd:end, :]
    
    # Find the row for this triangle
    row_number = findfirst(x -> x == it, all_curved_boundaries[:, 3])
    
    if row_number === nothing
        return false
    elseif Eₙ == 1
        # Check if edge 1 (between vertices 1 and 2) matches the boundary definition
        return (vn₁ == all_curved_boundaries[row_number, 1] && vn₂ == all_curved_boundaries[row_number, 2]) || 
               (vn₁ == all_curved_boundaries[row_number, 2] && vn₂ == all_curved_boundaries[row_number, 1])
    elseif Eₙ == 2
        # Check if edge 2 (between vertices 2 and 3) matches the boundary definition
        return (vn₂ == all_curved_boundaries[row_number, 1] && vn₃ == all_curved_boundaries[row_number, 2]) || 
               (vn₂ == all_curved_boundaries[row_number, 2] && vn₃ == all_curved_boundaries[row_number, 1])
    else
        # Check if edge 3 (between vertices 3 and 1) matches the boundary definition
        return (vn₃ == all_curved_boundaries[row_number, 1] && vn₁ == all_curved_boundaries[row_number, 2]) || 
               (vn₃ == all_curved_boundaries[row_number, 2] && vn₁ == all_curved_boundaries[row_number, 1])
    end
end

# Gets the boundary number for a specific triangle
# The boundary number is stored as a negative value in mesh.f[:, 4]
function get_boundary_number(mesh, it)
    i_bnd = findfirst(x -> x < 0, mesh.f[:, 4])
    all_curved_boundaries = mesh.f[i_bnd:end, :]
    row_number = findfirst(x -> x == it, all_curved_boundaries[:, 3])
    # Return the absolute value of the boundary number
    return -all_curved_boundaries[row_number, 4]
end

# Projects vertices of a mesh onto their corresponding boundaries
# distance_functions: array of functions defining the signed distance to each boundary
function project_vertex_to_boundary!(mesh::Mesh, distance_functions::Union{Nothing, Vector})
    if distance_functions !== nothing
        # Find the start of boundary definitions in mesh.f
        i_bnd = findfirst(x -> x < 0, mesh.f[:, 4])
        
        # Count number of unique boundary curves
        n_curves = length(Set(mesh.f[i_bnd:end, 4]))
        all_curved_faces = mesh.f[i_bnd:end, :]
        
        for i in 1:n_curves
            fd = distance_functions[i]
            
            # Get faces associated with boundary i
            curved_faces = all_curved_faces[all_curved_faces[:, 4] .== -i, :]
            
            # Collect unique nodes on this boundary
            unique_curved_nodes = Dict{Int, Nothing}()
            for node in curved_faces[:, 1:2]
                unique_curved_nodes[node] = nothing
            end
            
            # Project each boundary node onto the boundary curve
            for node in keys(unique_curved_nodes)
                node_coords = mesh.p[node, :]
                mesh.p[node, :] .= project_to_boundary(fd, node_coords, 0)
            end
        end
    end
end

"""
createdgnodes computes the coordinates of the dg nodes.
dgnodes=createnodes(mesh,fd)

   mesh:      mesh data structure
   fd:        distance function d(x,y)
   dgnodes:   triangle indices (nplx2xnt). the nodes on 
              the curved boundaries are projected to the
              true boundary using the distance function fd
"""
# Creates high-order nodes for a finite element mesh, handling curved boundaries
function createnodes(mesh, fd=nothing)
    npl = size(mesh.plocal, 1) 
    nt = size(mesh.t, 1)
    
    # First, project mesh vertices onto boundaries if needed
    project_vertex_to_boundary!(mesh, fd)
    dgnodes = zeros(npl, 2, nt)
    
    for it in axes(dgnodes, 3)
        # Get vertex indices for this triangle
        vn₁ = mesh.t[it, 1]
        vn₂ = mesh.t[it, 2]
        vn₃ = mesh.t[it, 3]
        
        # Get vertex coordinates
        v₁ = mesh.p[vn₁, :]
        v₂ = mesh.p[vn₂, :]
        v₃ = mesh.p[vn₃, :]
        
        # Check if this triangle has a curved edge
        iscurved_triangle = mesh.tcurved !== nothing && mesh.tcurved[it]
        
        # Loop through each local point within the triangle
        for ipl in axes(dgnodes, 1)
            # Get barycentric coordinates of this point
            λ = mesh.plocal[ipl, :]
            
            # Convert barycentric to Cartesian coordinates
            x = barycentric_to_cartesian(λ[2:3], v₁, v₂, v₃)
            
            # Store these coordinates in dgnodes
            dgnodes[ipl, :, it] .= x
            
            # Special handling for points on curved boundaries:
            # If all these conditions are met:
            # 1. We have distance functions
            # 2. This triangle has a curved edge
            # 3. The point is not a vertex
            # 4. The point is on an edge
            # 5. That edge is part of a curved boundary
            if fd !== nothing && iscurved_triangle && !isvertex(λ) && isedge(λ) && iscurvedboundary(λ, mesh, vn₁, vn₂, vn₃, it)
                # Get the boundary number for this curved edge
                fdn = get_boundary_number(mesh, it)
                
                # Project the point onto the curved boundary using the appropriate distance function
                x = project_to_boundary(fd[fdn], x, 0)
                
                # Update the node coordinates with the projected position
                dgnodes[ipl, :, it] .= x
            end
        end
    end
    
    # Create and return a new mesh with the same structure but using the computed high-order nodes
    return Mesh(mesh, dgnodes)
end