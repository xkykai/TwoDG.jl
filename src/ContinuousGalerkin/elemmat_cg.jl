
"""
elemmat_cg computes the elemental stiffness matrix and force vector.

    pcg:           element node coordinates
    mesh:         master structure
    source:       forcing function
    param:        kappa:   = diffusivity coefficient
                    c      = convective velocity
                    s      = source coefficient
    ae(npl,npl):  local element matrix (npl - nodes perelement)
    fe(npl):      local element force vector
"""
function elemmat_cg(pcg, master, source, param)
    npl = size(pcg, 1)
    A = zeros(npl, npl)
    κ = param.κ
    c = param.c
    s = param.s

    for k in eachindex(master.gwgh)
        ϕ = master.shap[:, 1, k]
        ∂ϕ∂ξ = master.shap[:, 2, k]
        ∂ϕ∂η = master.shap[:, 3, k]
        J = master.shap[:, 2:3, k]' * pcg
        detJ = det(J)
        invJ = inv(J)  # Inverse Jacobian for gradient transformation

        # Calculate diffusion term
        for i in axes(A, 1), j in axes(A, 2)
            ∂ϕᵢ∂x = ∂ϕ∂ξ[i] * invJ[1, 1] + ∂ϕ∂η[i] * invJ[1, 2]
            ∂ϕᵢ∂y = ∂ϕ∂ξ[i] * invJ[2, 1] + ∂ϕ∂η[i] * invJ[2, 2]
            ∂ϕⱼ∂x = ∂ϕ∂ξ[j] * invJ[1, 1] + ∂ϕ∂η[j] * invJ[1, 2]
            ∂ϕⱼ∂y = ∂ϕ∂ξ[j] * invJ[2, 1] + ∂ϕ∂η[j] * invJ[2, 2]
            A[i, j] += (∂ϕᵢ∂x * ∂ϕⱼ∂x + ∂ϕᵢ∂y * ∂ϕⱼ∂y) * κ * detJ * master.gwgh[k]
        end

        # Calculate convection term
        for i in axes(A, 1), j in axes(A, 2)
            ∂ϕᵢ∂x = ∂ϕ∂ξ[i] * invJ[1, 1] + ∂ϕ∂η[i] * invJ[1, 2]
            ∂ϕᵢ∂y = ∂ϕ∂ξ[i] * invJ[2, 1] + ∂ϕ∂η[i] * invJ[2, 2]
            cxϕⱼ = c[1] * ϕ[j]
            cyϕⱼ = c[2] * ϕ[j]
            A[i, j] += -(∂ϕᵢ∂x * cxϕⱼ + ∂ϕᵢ∂y * cyϕⱼ) * detJ * master.gwgh[k]
        end

        # Calculate reaction term
        for i in axes(A, 1), j in axes(A, 2)
            A[i, j] += ϕ[i] * ϕ[j] * s * detJ * master.gwgh[k]
        end

    end

    F = zeros(npl)
    for j in eachindex(master.gwgh)
        # Calculate Jacobian matrix at quadrature point
        J = master.shap[:, 2:3, j]' * pcg
        
        # Calculate determinant of Jacobian (represents area scaling factor)
        detJ = det(J)
        gauss_coordinate = vec(master.shap[:, 1, j]' * pcg)
        
        f = source(gauss_coordinate...)
        F .+= master.shap[:, 1, j] .* detJ .* f .* master.gwgh[j]
    end

    return A, F
end