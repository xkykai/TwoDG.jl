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
"""
function hdg_postprocess(master, mesh, master1, mesh1, uh, qh)
    # Extract necessary values
    porder = master.porder
    nps = porder + 1
    npl, _, nt = size(mesh.dgnodes)
    npl1, _, _ = size(mesh1.dgnodes)
    ng = length(master1.gwgh)
    @assert length(master.gwgh) == length(master1.gwgh)  # Ensure same number of quadrature points
    
    # In Julia, indexing starts at 1 (not 0 as in Python)
    perm = master.perm[:,:,1]
    
    # YOUR CODE HERE ....
    ustarh = zeros(npl1, 1, nt)
    r = zero(ustarh)
    
    # Volume integral (element contributions)
    shap1 = master1.shap[:, 1, :]     # Shape functions
    shapxi1 = master1.shap[:, 2, :]   # Shape function derivatives in ξ direction
    shapet1 = master1.shap[:, 3, :]   # Shape function derivatives in η direction
    shapxig1 = shapxi1 * Diagonal(master1.gwgh)  # Weighted shape function derivatives
    shapetg1 = shapet1 * Diagonal(master1.gwgh)

    for i in 1:nt
        curved_t = mesh1.tcurved[i]
        ng = length(master1.gwgh)
        
        qgx = master.shap[:, 1, :]' * qh[:, 1, i]  # qg = qh * shap
        qgy = master.shap[:, 1, :]' * qh[:, 2, i]  # qg = qh * shap

        if !curved_t
            # For straight-sided elements, geometric terms are constant
            # Calculate edge vectors of the element
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
            shapx1 = (shapxi1 * yet - shapet1 * yxi) / detJ
            shapy1 = (- shapxi1 * xet + shapet1 * xxi) / detJ

            r[:, 1, i] .-= (shapgx1 * qgx .+ shapgy1 * qgy) # r = ∂/∂x * qgx + ∂/∂y * qgy

            r[end, 1, i] = sum(master.shap[:, 1, :]' * uh[:, 1, i] .* master.gwgh .* detJ)
        else
            # For curved elements, geometric terms vary at each quadrature point
            coords = mesh1.dgnodes[:, :, i]
            detJ = zeros(ng)
            shapgx1 = zeros(npl1, ng)
            shapgy1 = zeros(npl1, ng)
            shapx1 = zeros(npl1, ng)
            shapy1 = zeros(npl1, ng)

            for j in eachindex(master1.gwgh)
                # Calculate Jacobian matrix at this quadrature point
                J = master1.shap[:, 2:3, j]' * coords
                invJ = inv(J)  # Inverse Jacobian needed for derivatives
                detJ[j] = det(J)
                
                # Transform reference derivatives to physical derivatives
                shap∇ = invJ * master1.shap[:, 2:3, j]'
                # Store weighted derivatives for integration
                shapgx1[:, j] .= shap∇[1, :] .* master1.gwgh[j] .* detJ[j]  # x-derivatives
                shapgy1[:, j] .= shap∇[2, :] .* master1.gwgh[j] .* detJ[j]  # y-derivatives
                shapx1[:, j] .= shap∇[1, :]  # x-derivatives without weights
                shapy1[:, j] .= shap∇[2, :]  # y-derivatives without weights
            end

            r[:, 1, i] .-= (shapgx1 * qgx .+ shapgy1 * qgy) # r = ∂/∂x * qgx + ∂/∂y * qgy

            r[end, 1, i] = sum(master.shap[:, 1, :]' * uh[:, 1, i] .* master.gwgh .* detJ)
        end

        M = shapx1 * Diagonal(master1.gwgh .* detJ) * shapx1' .+ shapy1 * Diagonal(master1.gwgh .* detJ) * shapy1' # Mass matrix

        M[end, :] .= sum(shap1 * Diagonal(master1.gwgh .* detJ), dims=2)  # Last row for boundary condition
        @info cond(M)

        ustarh[:, 1, i] .= M \ r[:, 1, i]  # Solve for postprocessed solution using mass matrix

        @info "u integral $(sum(master.shap[:, 1, :]' * uh[:, 1, i] .* master.gwgh .* detJ))"
        @info "u⋆ integral $(sum(master1.shap[:, 1, :]' * ustarh[:, 1, i] .* master1.gwgh .* detJ))"
    end

    return ustarh
end