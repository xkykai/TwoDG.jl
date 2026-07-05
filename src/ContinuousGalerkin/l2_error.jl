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
- `exact`: Exact solution function of the coordinates, `(x, y) -> value` in
  2D, `(x, y, z) -> value` in 3D

# Returns
- `‖uh - exact‖_{L²(Ω)}`
"""
function l2error(mesh, uh::AbstractMatrix, exact::Function)
    # Use high order integration to calculate the error
    mst = Master(mesh, 4 * mesh.porder)
    Dim = size(mesh.dgnodes, 2)
    shap = mst.shap[:, 1, :]                            # values (npl, ng)
    derivs = ntuple(b -> mst.shap[:, b + 1, :], Dim)    # ∂/∂ξ_b tables
    ng = size(shap, 2)

    err2 = 0.0
    J = Matrix{Float64}(undef, Dim, Dim)
    @inbounds for i in 1:size(mesh.t, 1)
        dg = @view mesh.dgnodes[:, :, i]

        # per-quadrature-point Jacobian determinant of x(ξ)
        dXdξ = ntuple(b -> derivs[b]' * dg, Dim)        # each (ng, Dim)
        jac = map(1:ng) do g
            for b in 1:Dim, a in 1:Dim
                J[a, b] = dXdξ[b][g, a]
            end
            det(J)
        end

        # Evaluate solution and exact solution at quadrature points
        ug = shap' * view(uh, :, i)
        quad_points = shap' * dg
        ugexact = exact.(ntuple(d -> view(quad_points, :, d), Dim)...)
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
