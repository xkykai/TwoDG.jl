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
        for j in 1:size(master.gpts, 1)
            J = shapefunction_2d[:, 2:3, j]' * element_coordinates
            detJ = det(J)
            area1 += sum(master.gwgh[j] * shapefunction_2d[:, 1, j] * detJ)
        end
    end

    return area1
end