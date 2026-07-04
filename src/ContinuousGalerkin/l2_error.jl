using TwoDG.Masters: Master
using LinearAlgebra

"""
    l2error(mesh, uh, exact)

L2 norm of the error between a numerical solution and an exact solution,
computed with high-order (4p) quadrature.

# Arguments
- `mesh`: Mesh structure
- `uh`: Scalar field with local (DG) numbering — either `(npl, nt)` or a
  single-component `(npl, 1, nt)` array
- `exact`: Exact solution function `(x, y) -> value`

# Returns
- `‖uh - exact‖_{L²(Ω)}`
"""
function l2error(mesh, uh::AbstractMatrix, exact::Function)
    # Use high order integration to calculate the error
    mst = Master(mesh, 4 * mesh.porder)

    err2 = 0.0
    @inbounds for i in 1:size(mesh.t, 1)
        @views begin
            shap = mst.shap[:, 1, :]    # values
            shapxi = mst.shap[:, 2, :]  # d/dxi
            shapet = mst.shap[:, 3, :]  # d/deta
            dg = mesh.dgnodes[:, :, i]
        end

        # Metric terms and Jacobian determinant
        xxi = shapxi' * dg[:, 1]
        xet = shapet' * dg[:, 1]
        yxi = shapxi' * dg[:, 2]
        yet = shapet' * dg[:, 2]
        jac = xxi .* yet - xet .* yxi

        # Evaluate solution and exact solution at quadrature points
        ug = shap' * view(uh, :, i)
        quad_points = shap' * dg
        ugexact = exact.(quad_points[:, 1], quad_points[:, 2])
        ugerror = ug - ugexact

        err2 += sum((mst.gwgh .* jac) .* ugerror .^ 2)
    end

    return sqrt(err2)
end

function l2error(mesh, uh::AbstractArray{<:Any, 3}, exact::Function)
    size(uh, 2) == 1 || throw(ArgumentError(
        "l2error expects a scalar field; pass one component, e.g. `u[:, c, :]`"))
    return l2error(mesh, reshape(uh, size(uh, 1), :), exact)
end

