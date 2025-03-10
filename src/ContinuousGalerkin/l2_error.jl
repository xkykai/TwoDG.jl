using TwoDG.Masters: Master
using LinearAlgebra

function l2_error(mesh, uh, exact)
    """
    l2_error calculates the l2 norm of the error (squared)
    mesh:         mesh structure
    uh:           scalar variable with local numbering
    exact:        exact solution function
    return:       l2 norm of the error squared
    """
    # Use high order integration to calculate the error
    mst = Master(mesh, 4*mesh.porder)
   
    l2error = 0.0
    for i in 1:size(mesh.t, 1)
        shap = mst.shap[:,1,:]
        shapxi = mst.shap[:,2,:]
        shapet = mst.shap[:,3,:]
        dg = mesh.dgnodes[:,:,i]
        xxi = dg[:,1]' * shapxi
        xet = dg[:,1]' * shapet
        yxi = dg[:,2]' * shapxi
        yet = dg[:,2]' * shapet
        jac = xxi .* yet - xet .* yxi
   
        ug = shap' * uh[:,i]
        quad_points = shap' * dg
        ugexact = exact.(quad_points[:, 1], quad_points[:, 2])     # exact solution at quadrature points
        ugerror = ug - ugexact
        l2error += ugerror' * Diagonal(mst.gwgh .* jac) * ugerror
    end
    return l2error
end