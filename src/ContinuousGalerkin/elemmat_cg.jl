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
    npl = size(pcg, 1)  # Number of nodes per element
    κ = param.κ
    c = param.c
    s = param.s
    A = zeros(npl, npl)  # Initialize elemental stiffness matrix
    F = zeros(npl)       # Initialize elemental force vector
    for k in eachindex(master.gwgh)
        # Get shape functions and their derivatives at quadrature point k
        ϕ = master.shap[:, 1, k]     # Shape functions
        ∂ϕ∂ξ = master.shap[:, 2, k]  # Derivatives w.r.t. ξ (reference coordinate)
        ∂ϕ∂η = master.shap[:, 3, k]  # Derivatives w.r.t. η (reference coordinate)
        
        # Compute Jacobian for mapping between reference and physical elements
        J = master.shap[:, 2:3, k]' * pcg
        detJ = det(J)  # Determinant for area/volume scaling
        invJ = inv(J)  # Inverse Jacobian for gradient transformation
        
        # Calculate contributions to the stiffness matrix
        for i in axes(A, 1), j in axes(A, 2)
            # Transform derivatives from reference to physical coordinates using chain rule
            ∂ϕᵢ∂x = ∂ϕ∂ξ[i] * invJ[1, 1] + ∂ϕ∂η[i] * invJ[1, 2]
            ∂ϕᵢ∂y = ∂ϕ∂ξ[i] * invJ[2, 1] + ∂ϕ∂η[i] * invJ[2, 2]
            ∂ϕⱼ∂x = ∂ϕ∂ξ[j] * invJ[1, 1] + ∂ϕ∂η[j] * invJ[1, 2]
            ∂ϕⱼ∂y = ∂ϕ∂ξ[j] * invJ[2, 1] + ∂ϕ∂η[j] * invJ[2, 2]
            
            # Pre-compute convective terms with shape functions
            cxϕⱼ = c[1] * ϕ[j]  # x-component of convection with basis function j
            cyϕⱼ = c[2] * ϕ[j]  # y-component of convection with basis function j
            
            # Add diffusion term: κ∫(∇ϕᵢ·∇ϕⱼ)dΩ
            A[i, j] += (∂ϕᵢ∂x * ∂ϕⱼ∂x + ∂ϕᵢ∂y * ∂ϕⱼ∂y) * κ * detJ * master.gwgh[k]
            
            # Add convection term: -∫(c·∇ϕᵢ)ϕⱼdΩ
            A[i, j] += -(∂ϕᵢ∂x * cxϕⱼ + ∂ϕᵢ∂y * cyϕⱼ) * detJ * master.gwgh[k]
            
            # Add reaction term: s∫(ϕᵢϕⱼ)dΩ
            A[i, j] += ϕ[i] * ϕ[j] * s * detJ * master.gwgh[k]
        end
        
        # Map quadrature point to physical coordinates for source evaluation
        gauss_coordinate = vec(master.shap[:, 1, k]' * pcg)
       
        # Evaluate source function at current quadrature point
        f = source(gauss_coordinate...)
        
        # Add source contribution to force vector: ∫f·ϕᵢdΩ
        F .+= master.shap[:, 1, k] .* detJ .* f .* master.gwgh[k]
    end
    return A, F
end