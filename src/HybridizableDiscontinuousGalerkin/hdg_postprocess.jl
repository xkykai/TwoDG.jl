"""
    hdg_postprocess(master, mesh, master1, mesh1, uh, qh)

Postprocesses the HDG solution to obtain a better solution.

# Arguments
- `master`: master structure of porder
- `mesh`: mesh structure of porder
- `master1`: master structure of porder+1
- `mesh1`: mesh structure of porder+1
- `uh`: approximate scalar variable
- `qh`: approximate flux

# Returns
- `ustarh`: postprocessed scalar variable

# HDG Postprocessing
The Hybridizable Discontinuous Galerkin (HDG) method allows for local element-by-element
postprocessing to increase the accuracy of the solution. This function solves the local
postprocessing problem:

1. Find u* in P_{p+1}(K) such that:
   - ∇u* = -qh in each element K
   - The average of u* equals the average of uh in each K

This typically yields a solution with p+2 order of convergence for diffusion problems.
"""
function hdg_postprocess(master, mesh, master1, mesh1, uh, qh)
    # Extract necessary values
    porder = master.porder
    nps = porder + 1
    npl, _, nt = size(mesh.dgnodes)    # npl: nodes per element, nt: number of elements
    npl1, _, _ = size(mesh1.dgnodes)   # npl1: nodes per element for p+1 approximation
    ng = length(master1.gwgh)          # ng: number of quadrature points
    @assert length(master.gwgh) == length(master1.gwgh)  # Ensure same number of quadrature points
    
    # In Julia, indexing starts at 1 (not 0 as in Python)
    perm = master.perm[:,:,1]          # Permutation for element boundaries
    
    # Initialize the postprocessed solution and right-hand side vector
    ustarh = zeros(npl1, 1, nt)        # Postprocessed solution of order p+1
    r = zero(ustarh)                   # Right-hand side for local problems
    
    # Precompute shape functions and derivatives for integration
    # These are defined on the reference element
    shap1 = master1.shap[:, 1, :]      # Shape functions
    shapxi1 = master1.shap[:, 2, :]    # Shape function derivatives in ξ direction
    shapet1 = master1.shap[:, 3, :]    # Shape function derivatives in η direction
    # Pre-weight derivatives with quadrature weights for integration
    shapxig1 = shapxi1 * Diagonal(master1.gwgh)  # Weighted shape function derivatives
    shapetg1 = shapet1 * Diagonal(master1.gwgh)

    # Process each element in parallel
    Threads.@threads for i in 1:nt
        curved_t = mesh1.tcurved[i]    # Check if element is curved
        ng = length(master1.gwgh)
        
        # Project flux vector onto quadrature points for current element
        qgx = master.shap[:, 1, :]' * qh[:, 1, i]  # x-component of flux at quadrature points
        qgy = master.shap[:, 1, :]' * qh[:, 2, i]  # y-component of flux at quadrature points

        if !curved_t
            # For straight-sided elements, geometric terms are constant across the element
            
            # Calculate edge vectors of the element to form Jacobian matrix components
            xxi = mesh1.p[mesh1.t[i, 2], 1] - mesh1.p[mesh1.t[i, 1], 1]  # ∂x/∂ξ component
            xet = mesh1.p[mesh1.t[i, 3], 1] - mesh1.p[mesh1.t[i, 1], 1]  # ∂x/∂η component
            yxi = mesh1.p[mesh1.t[i, 2], 2] - mesh1.p[mesh1.t[i, 1], 2]  # ∂y/∂ξ component
            yet = mesh1.p[mesh1.t[i, 3], 2] - mesh1.p[mesh1.t[i, 1], 2]  # ∂y/∂η component
            
            # Determinant of Jacobian matrix (twice the element area)
            detJ = xxi * yet - xet * yxi
            invJ = inv([xxi xet; yxi yet])  # Inverse Jacobian matrix
            
            # Convert reference derivatives to physical derivatives using chain rule
            # These include quadrature weights and Jacobian for integration
            shapgx1 =   shapxig1 * yet - shapetg1 * yxi  # ∂/∂x = ∂/∂ξ * ∂ξ/∂x + ∂/∂η * ∂η/∂x
            shapgy1 = - shapxig1 * xet + shapetg1 * xxi  # ∂/∂y = ∂/∂ξ * ∂ξ/∂y + ∂/∂η * ∂η/∂y
            shapx1 = (shapxi1 * yet - shapet1 * yxi) / detJ   # Unweighted x-derivatives
            shapy1 = (- shapxi1 * xet + shapet1 * xxi) / detJ # Unweighted y-derivatives

            # Build right-hand side using the HDG constraint: ∇u* = -qh
            # This enforces that the gradient of postprocessed solution matches the computed flux
            r[:, 1, i] .-= (shapgx1 * qgx .+ shapgy1 * qgy) # r = ∫ ∇v·qh dK

            # Set constraint: average of u* equals average of uh (last equation)
            # This provides the necessary constraint to make the problem well-posed
            r[end, 1, i] = sum(master.shap[:, 1, :]' * uh[:, 1, i] .* master.gwgh .* detJ)
        else
            # For curved elements, geometric terms vary at each quadrature point
            coords = mesh1.dgnodes[:, :, i]  # Physical coordinates of nodes in this element
            detJ = zeros(ng)                 # Jacobian determinant at each quadrature point
            shapgx1 = zeros(npl1, ng)        # Weighted x-derivatives for integration
            shapgy1 = zeros(npl1, ng)        # Weighted y-derivatives for integration
            shapx1 = zeros(npl1, ng)         # Unweighted x-derivatives for mass matrix
            shapy1 = zeros(npl1, ng)         # Unweighted y-derivatives for mass matrix

            for j in eachindex(master1.gwgh)
                # Calculate Jacobian matrix at this quadrature point by mapping reference to physical space
                J = master1.shap[:, 2:3, j]' * coords
                invJ = inv(J)               # Inverse Jacobian needed for derivatives
                detJ[j] = det(J)            # Determinant for integration scaling
                
                # Transform reference derivatives to physical derivatives: ∇ξ → ∇x
                shap∇ = invJ * master1.shap[:, 2:3, j]'
                
                # Store weighted derivatives for integration
                shapgx1[:, j] .= shap∇[1, :] .* master1.gwgh[j] .* detJ[j]  # x-derivatives
                shapgy1[:, j] .= shap∇[2, :] .* master1.gwgh[j] .* detJ[j]  # y-derivatives
                shapx1[:, j] .= shap∇[1, :]  # x-derivatives without weights
                shapy1[:, j] .= shap∇[2, :]  # y-derivatives without weights
            end

            # Build right-hand side (same as for straight elements, but with point-wise transforms)
            r[:, 1, i] .-= (shapgx1 * qgx .+ shapgy1 * qgy) # r = ∫ ∇v·qh dK

            # Set constraint: average of u* equals average of uh
            r[end, 1, i] = sum(master.shap[:, 1, :]' * uh[:, 1, i] .* master.gwgh .* detJ)
        end

        # Construct mass matrix for Poisson-like problem: ∫ ∇u*·∇v = ∫ -qh·∇v
        # This creates the stiffness matrix ∫ ∇φ_i·∇φ_j dK where φ are basis functions
        M = shapx1 * Diagonal(master1.gwgh .* detJ) * shapx1' .+ 
            shapy1 * Diagonal(master1.gwgh .* detJ) * shapy1' 

        # Replace last row with constraint: average of u* equals average of uh
        # This enforces the uniqueness of the postprocessed solution
        M[end, :] .= sum(shap1 * Diagonal(master1.gwgh .* detJ), dims=2)  

        # Solve local system to find postprocessed solution
        ustarh[:, 1, i] .= M \ r[:, 1, i]  # Solve: M * u* = r
    end

    return ustarh
end