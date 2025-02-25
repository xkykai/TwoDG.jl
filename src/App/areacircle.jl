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
    mesh = mkmesh_circle(siz, porder)
    master = Master(mesh)

    area1 = 0.

    dgnodes = mesh.dgnodes

    shapefunction_2d = master.shap

    for i in axes(dgnodes, 3)
        element_coordinates = @view dgnodes[:, :, i]
        for j in axes(master.gpts, 1)
            J = shapefunction_2d[:, 2:3, j]' * element_coordinates
            detJ = det(J)
            area1 += sum(master.gwgh[j] * shapefunction_2d[:, 1, j] * detJ)
        end
    end

    area2 = 0.

    shapefunction_1d = master.sh1d

    for i in axes(mesh.f, 1)
        if mesh.f[i, 4] == -1
            it = mesh.f[i, 3]
            face_start = mesh.f[i, 1]
            face_location = findfirst(x -> x == face_start, mesh.t[it, :])
            face_number = mod1(face_location - 1, 3)

            node_numbers = master.perm[:, face_number, 1]
            element_coordinates = @view dgnodes[node_numbers, :, it]
            
            for j in eachindex(master.gw1d)
                τ = shapefunction_1d[:, 2, j]' * element_coordinates
                τ_norm = sqrt(sum(τ.^2))
                τ ./= τ_norm 

                gauss_coordinate = shapefunction_1d[:, 1, j]' * element_coordinates
                P = gauss_coordinate ./ 2
                n = [τ[2], -τ[1]]

                area2 -= sum(master.gw1d[j] * shapefunction_1d[:, 1, j] * τ_norm * dot(P, n))
            end

        end
    end

    perim = 0.

    for i in axes(mesh.f, 1)
        if mesh.f[i, 4] == -1
            it = mesh.f[i, 3]
            face_start = mesh.f[i, 1]
            face_location = findfirst(x -> x == face_start, mesh.t[it, :])
            face_number = mod1(face_location - 1, 3)

            node_numbers = master.perm[:, face_number, 1]
            element_coordinates = @view dgnodes[node_numbers, :, it]
            
            for j in eachindex(master.gw1d)
                τ = shapefunction_1d[:, 2, j]' * element_coordinates
                τ_norm = sqrt(sum(τ.^2))
                τ ./= τ_norm 

                perim += sum(master.gw1d[j] * shapefunction_1d[:, 1, j] * τ_norm)
            end

        end
    end

    return area1, area2, perim
end