using TwoDG.Masters: Master
using LinearAlgebra

"""
    l2_error(mesh, uh, exact)

Calculate the L2 norm of the error (squared) between a numerical solution and an exact solution.

# Arguments
- `mesh`: Mesh structure
- `uh`: Scalar variable with local numbering
- `exact`: Exact solution function that takes coordinates and returns values

# Returns
- The L2 norm of the error squared
"""
function l2_error(mesh, uh::AbstractArray, exact::Function)
    # Use high order integration to calculate the error
    mst = Master(mesh, 4*mesh.porder)
    
    l2error = 0.0
    @inbounds for i in 1:size(mesh.t, 1)
        # Extract shape functions and derivatives
        @views begin
            shap = mst.shap[:, 1, :]    # Values (1-based indexing)
            shapxi = mst.shap[:, 2, :]  # d/dxi
            shapet = mst.shap[:, 3, :]  # d/deta
            dg = mesh.dgnodes[:, :, i]
        end
        
        # Compute metric terms
        xxi = shapxi' * dg[:,1]
        xet = shapet' * dg[:,1]
        yxi = shapxi' * dg[:,2]
        yet = shapet' * dg[:,2]
        
        # Compute Jacobian determinant
        jac = xxi .* yet - xet .* yxi
        
        # Evaluate solution and exact solution at quadrature points
        ug = shap' * view(uh, :, i)
        quad_points = shap' * dg
        ugexact = exact.(quad_points[:, 1], quad_points[:, 2])     # exact solution at quadrature points
        ugerror = ug - ugexact
        
        # Compute weighted error using element-wise operations
        l2error += sum((mst.gwgh .* jac) .* ugerror.^2)
    end
    
    return l2error
end