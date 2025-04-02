using TwoDG.Meshes: Mesh, mkmesh_circle
using TwoDG.Masters: Master, shape2d
using LinearAlgebra

"""    
areacircle calcualte the area and perimeter of a unit circle
[area,perim]=areacircle(sp,porder)

   siz:       desired element size 
   porder:    polynomial order of approximation (default=1)
   area1:     area of the circle (π)
   area2:     area of the circle (π)
   perim:     perimeter of the circumference (2π)
"""
function areacircle(siz, porder)
    # Create a circular mesh with specified element size and polynomial order
    mesh = mkmesh_circle(siz, porder)
    
    # Create a master element for reference shape and quadrature
    master = Master(mesh)
    
    # Initialize area calculation using volume integration
    area1 = 0.
    dgnodes = mesh.dgnodes
    shapefunction_2d = master.shap
    
    # Loop over all elements in the mesh
    for i in axes(dgnodes, 3)
        element_coordinates = @view dgnodes[:, :, i]
        
        # Loop over all Gaussian quadrature points
        for j in axes(master.gpts, 1)
            # Calculate Jacobian matrix at quadrature point
            J = shapefunction_2d[:, 2:3, j]' * element_coordinates
            
            # Calculate determinant of Jacobian (represents area scaling factor)
            detJ = det(J)
            
            # Accumulate area contribution: quadrature weight * shape function * determinant
            area1 += sum(master.gwgh[j] * shapefunction_2d[:, 1, j] * detJ)
        end
    end
    
    # Alternative area calculation using boundary integration (Green's theorem)
    area2 = 0.
    shapefunction_1d = master.sh1d
    
    # Loop over all faces in the mesh
    for i in axes(mesh.f, 1)
        # Check if face is on the boundary (indicated by -1)
        if mesh.f[i, 4] == -1
            it = mesh.f[i, 3]  # Element containing this face
            face_start = mesh.f[i, 1]  # Starting node of the face
            
            # Find the local face number within the element
            face_location = findfirst(x -> x == face_start, mesh.t[it, :])
            face_number = mod1(face_location - 1, 3)
            
            # Get node numbers for this face using permutation mapping
            node_numbers = master.perm[:, face_number, 1]
            element_coordinates = @view dgnodes[node_numbers, :, it]
           
            # Loop over boundary quadrature points
            for j in eachindex(master.gw1d)
                # Calculate tangent vector at quadrature point
                τ = shapefunction_1d[:, 2, j]' * element_coordinates
                τ_norm = sqrt(sum(τ.^2))
                τ ./= τ_norm
                
                # Calculate physical coordinate of quadrature point
                gauss_coordinate = shapefunction_1d[:, 1, j]' * element_coordinates
                
                # Point on the boundary (divided by 2 for centered coordinates)
                P = gauss_coordinate ./ 2
                
                # Normal vector (perpendicular to tangent)
                n = [τ[2], -τ[1]]
                
                # Green's theorem for area calculation: ∮(x·n)ds
                # Negative sign because boundary integration is clockwise
                area2 -= sum(master.gw1d[j] * shapefunction_1d[:, 1, j] * τ_norm * dot(P, n))
            end
        end
    end
    
    # Calculate perimeter using boundary integration
    perim = 0.
    
    # Loop over all faces in the mesh
    for i in axes(mesh.f, 1)
        # Check if face is on the boundary
        if mesh.f[i, 4] == -1
            it = mesh.f[i, 3]  # Element containing this face
            face_start = mesh.f[i, 1]  # Starting node of the face
            
            # Find the local face number within the element
            face_location = findfirst(x -> x == face_start, mesh.t[it, :])
            face_number = mod1(face_location - 1, 3)
            
            # Get node numbers for this face using permutation mapping
            node_numbers = master.perm[:, face_number, 1]
            element_coordinates = @view dgnodes[node_numbers, :, it]
           
            # Loop over boundary quadrature points
            for j in eachindex(master.gw1d)
                # Calculate tangent vector and its norm at quadrature point
                τ = shapefunction_1d[:, 2, j]' * element_coordinates
                τ_norm = sqrt(sum(τ.^2))
                τ ./= τ_norm
                
                # Accumulate perimeter: quadrature weight * shape function * tangent norm
                perim += sum(master.gw1d[j] * shapefunction_1d[:, 1, j] * τ_norm)
            end
        end
    end
    
    return area1, area2, perim
end