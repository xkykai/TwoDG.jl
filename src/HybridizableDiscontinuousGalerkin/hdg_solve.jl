using LinearAlgebra
using SparseArrays
# BLAS.set_num_threads(1)

# HDG Method Overview:
# The Hybridizable Discontinuous Galerkin method works by introducing a hybrid variable (uhat)
# that represents the trace of the solution on element faces. This enables static condensation
# where we solve a global system only for uhat, then recover local solutions (uh, qh) element-by-element.
# This approach is more efficient than standard DG for higher-order approximations.

"""
    localprob(dg, master, m, source, param)

Solves the local convection-diffusion problems for the HDG method.

# Arguments
- `dg`: DG nodes
- `master`: Master element structure
- `m`: Values of uhat at element edges
- `source`: Source term function or nothing
- `param`: Dictionary with parameters `:kappa` (diffusivity) and `:c` (convective velocity)

# Returns
- `umf`: Local solution uh
- `qmf`: Local solution qh
"""
@inline function localprob(dg, master, m, source, param)
    # Extract parameters
    kappa = param[:kappa]
    c = param[:c]
    taud = kappa
    
    porder = master.porder
    nps = porder + 1
    ncol = size(m, 2)
    npl = size(dg, 1)
    
    # Pre-allocate all solution arrays
    qmf = zeros(npl, 2, ncol)
    umf = zeros(npl, ncol)
    
    # Pre-allocate flux and source terms - reused across iterations
    Fx = zeros(npl, ncol)
    Fy = zeros(npl, ncol)
    Fu = zeros(npl, ncol)
    
    # Cache frequently accessed views - avoid repeated view creation
    perm = @view master.perm[:,:,1]
    shap = @view master.shap[:, 1, :]
    shapxi = @view master.shap[:, 2, :]
    shapet = @view master.shap[:, 3, :]
    sh1d = @view master.sh1d[:,1,:]
    sh1d_deriv = @view master.sh1d[:,2,:]
    
    # Pre-allocate edge computation buffers
    xxi_edge = zeros(size(sh1d_deriv, 2))
    yxi_edge = zeros(size(sh1d_deriv, 2))
    dsdxi = zeros(size(sh1d_deriv, 2))
    nl = zeros(size(sh1d_deriv, 2), 2)
    cnl = zeros(size(sh1d_deriv, 2))
    tauc = zeros(size(sh1d_deriv, 2))
    tau = zeros(size(sh1d_deriv, 2))
    
    # Pre-compute physical derivatives with minimal allocations
    dgx = @view dg[:,1]
    dgy = @view dg[:,2]
    
    # Use mul! for matrix multiplications where possible
    xxi = shapxi' * dgx  # ∂x/∂ξ
    xet = shapet' * dgx  # ∂x/∂η
    yxi = shapxi' * dgy  # ∂y/∂ξ 
    yet = shapet' * dgy  # ∂y/∂η
    
    # Compute Jacobian determinant
    jac = xxi .* yet - xet .* yxi
    
    # Pre-compute diagonal matrices for efficiency
    gwgh_diag = Diagonal(master.gwgh)
    yet_diag = Diagonal(yet)
    yxi_diag = Diagonal(yxi)
    xet_diag = Diagonal(xet)
    xxi_diag = Diagonal(xxi)
    
    # Shape derivatives in physical space
    shapxig = shapxi * gwgh_diag
    shapetg = shapet * gwgh_diag
    
    # Use pre-allocated matrices for derivatives
    shapx = shapxig * yet_diag - shapetg * yxi_diag
    shapy = -shapxig * xet_diag + shapetg * xxi_diag
    
    # Mass matrix computation
    gwgh_jac = master.gwgh .* jac
    gwgh_jac_diag = Diagonal(gwgh_jac)
    
    # Mass matrix and coupling matrices - cache friendly computation
    M = (shap * gwgh_jac_diag * shap') / kappa
    
    # Use direct factorization instead of inverse for better numerical stability
    M_fact = lu(M)
    
    # Coupling matrices
    Cx = shap * shapx'
    Cy = shap * shapy'
    
    # Convection operator
    D = -c[1] .* Cx' - c[2] .* Cy'
    
    # Process source term if provided
    if source isa Function
        # Reuse existing arrays for computing source
        pg = shap' * dg
        src = source(pg)
        mul!(Fu, shap * gwgh_jac_diag, src)
    end
    
    # First loop: Add stabilization terms to diffusion matrix
    @views for s in 1:3
        perm_s = perm[:, s]
        
        # Calculate edge terms in-place
        mul!(xxi_edge, sh1d_deriv', dg[perm_s, 1])
        mul!(yxi_edge, sh1d_deriv', dg[perm_s, 2])
        
        # Compute edge metrics
        @. dsdxi = sqrt(xxi_edge^2 + yxi_edge^2)
        @. nl[:,1] = yxi_edge/dsdxi
        @. nl[:,2] = -xxi_edge/dsdxi
        
        # Compute stabilization parameters
        @. cnl = c[1] * nl[:,1] + c[2] * nl[:,2]
        @. tauc = abs(cnl)
        @. tau = taud + tauc
        
        # Pre-compute weighted quadrature values for better cache efficiency
        tau_quad = master.gw1d .* dsdxi .* tau
        
        # Add stabilization matrix in a cache-friendly way
        # Compute edge stabilization matrix once per edge
        edge_stab = sh1d * Diagonal(tau_quad) * sh1d'
        
        # Add to diffusion matrix
        D[perm_s, perm_s] .+= edge_stab
    end
    
    # Pre-allocate edge matrices to avoid repeated allocations in loop
    edge_matrix_x = zeros(nps, nps)
    edge_matrix_y = zeros(nps, nps)
    edge_matrix_u = zeros(nps, nps)
    tmp_result = zeros(nps)  # Pre-allocate buffer for temporary results
    
    # Second loop: Edge contributions to right-hand side
    @views for s in 1:3
        perm_s = perm[:, s]
        
        # Calculate edge terms more efficiently by reusing previous arrays
        mul!(xxi_edge, sh1d_deriv', dg[perm_s, 1])
        mul!(yxi_edge, sh1d_deriv', dg[perm_s, 2])
        
        # Compute metrics in-place
        @. dsdxi = sqrt(xxi_edge^2 + yxi_edge^2)
        @. nl[:,1] = yxi_edge/dsdxi
        @. nl[:,2] = -xxi_edge/dsdxi
        @. cnl = c[1] * nl[:,1] + c[2] * nl[:,2]
        @. tauc = abs(cnl)
        @. tau = taud + tauc
        
        # Pre-compute weighted quadrature values
        gw1d_dsdxi = master.gw1d .* dsdxi
        
        # Pre-compute edge matrices once per edge for better cache locality
        mul!(edge_matrix_x, sh1d * Diagonal(gw1d_dsdxi .* nl[:,1]), sh1d')
        mul!(edge_matrix_y, sh1d * Diagonal(gw1d_dsdxi .* nl[:,2]), sh1d')
        mul!(edge_matrix_u, sh1d * Diagonal(gw1d_dsdxi .* (cnl .- tau)), sh1d')
        
        # Process each column of the trace variable
        for icol in 1:ncol
            ml = m[(s-1)*nps+1:s*nps, icol]
            
            # Compute result in temporary buffer first
            mul!(tmp_result, edge_matrix_x, ml)
            Fx[perm_s, icol] .-= tmp_result
            
            mul!(tmp_result, edge_matrix_y, ml)
            Fy[perm_s, icol] .-= tmp_result
            
            mul!(tmp_result, edge_matrix_u, ml)
            Fu[perm_s, icol] .-= tmp_result
        end
    end
    
    # Use factorization instead of inverse for better performance
    # Pre-allocate arrays for repeated operations
    M1Fx = zeros(size(Fx))
    M1Fy = zeros(size(Fy))
    
    # Solve M*M1Fx = Fx instead of computing inverse
    for j in 1:ncol
        ldiv!(view(M1Fx, :, j), M_fact, view(Fx, :, j))
        ldiv!(view(M1Fy, :, j), M_fact, view(Fy, :, j))
    end
    
    # Compute system matrix with minimal allocations
    # This is a critical computation affecting performance
    CxM1Cx = Cx * (M_fact \ Cx')
    CyM1Cy = Cy * (M_fact \ Cy')
    system_matrix = D + CxM1Cx + CyM1Cy
    
    # Pre-allocate RHS vector
    system_rhs = zeros(size(Fu))
    
    # Compute RHS with minimal allocations
    for j in 1:ncol
        # Use BLAS for matrix-vector operations
        BLAS.gemv!('N', -1.0, Cx, view(M1Fx, :, j), 1.0, view(Fu, :, j))
        BLAS.gemv!('N', -1.0, Cy, view(M1Fy, :, j), 1.0, view(Fu, :, j))
        
        # Solve for u component-wise for better cache locality
        ldiv!(view(umf, :, j), lu(system_matrix), view(Fu, :, j))
    end
    
    # Recover flux q with minimal allocations
    # Pre-compute Cx'*umf for reuse
    Cx_umf = zeros(size(Fx))
    Cy_umf = zeros(size(Fx))
    
    # Use BLAS for better performance
    for j in 1:ncol
        BLAS.gemv!('T', 1.0, Cx, view(umf, :, j), 0.0, view(Cx_umf, :, j))
        BLAS.gemv!('T', 1.0, Cy, view(umf, :, j), 0.0, view(Cy_umf, :, j))
        
        # Solve for flux components directly
        ldiv!(view(qmf, :, 1, j), M_fact, view(Cx_umf, :, j))
        ldiv!(view(qmf, :, 2, j), M_fact, view(Cy_umf, :, j))
        
        # Add M1Fx and M1Fy components
        @. qmf[:, 1, j] += M1Fx[:, j]
        @. qmf[:, 2, j] += M1Fy[:, j]
    end
    
    return umf, qmf
end

"""
    elemmat_hdg(dg, master, source, param)

Calculates the element and force vectors for the HDG method.

# Arguments
- `dg`: DG nodes
- `master`: Master element structure
- `source`: Source term function or nothing
- `param`: Dictionary with parameters `:kappa` (diffusivity) and `:c` (convective velocity)

# Returns
- `ae`: Element matrix
- `fe`: Element force vector
"""
@inline function elemmat_hdg(dg, master, source, param)
    nps = master.porder + 1

    kappa = param[:kappa]
    c = param[:c]
    taud = param[:taud]  # Stabilization parameter

    # Create identity matrix for local problem - use sparse representation for large systems
    mu = I(3*nps)
    
    # Solve local problems with unit values of uhat
    um0, qm0 = localprob(dg, master, mu, nothing, param)

    # Zero matrix for force vector computation - reuse memory
    m_zero = zeros(3*nps, 1)
    u0f, q0f = localprob(dg, master, m_zero, source, param)

    # Initialize element matrix and force vector
    ae = zeros(3*nps, 3*nps)
    fe = zeros(3*nps)
    
    # Cache frequently accessed views
    perm = @view master.perm[:,:,1]
    sh1d = @view master.sh1d[:,1,:]
    sh1d_deriv = @view master.sh1d[:,2,:]
    
    # Pre-compute transpose of shape functions
    sh1d_t = sh1d'
    
    # Pre-allocate buffers for edge computations
    xxi_edge = zeros(size(sh1d_deriv, 2))
    yxi_edge = zeros(size(sh1d_deriv, 2))
    dsdxi = zeros(size(sh1d_deriv, 2))
    nl = zeros(size(sh1d_deriv, 2), 2)
    cnl = zeros(size(sh1d_deriv, 2))
    tauc = zeros(size(sh1d_deriv, 2))
    tau = zeros(size(sh1d_deriv, 2))
    
    # Pre-allocate arrays for inner loops
    nul = zeros(nps, 1)
    nug = zeros(size(sh1d_t, 1), 1)
    ug = zeros(size(sh1d_t, 1), 1)
    qgx = zeros(size(sh1d_t, 1), 1)
    qgy = zeros(size(sh1d_t, 1), 1)
    qh = zeros(size(sh1d_t, 1), 1)
    qhi = zeros(size(sh1d_t, 1), 1)
    
    @views for s in 1:3  # Loop over the 3 edges of the triangle
        perm_s = perm[:, s]
        
        # Calculate normal vectors and Jacobian - reuse pre-allocated arrays
        mul!(xxi_edge, sh1d_deriv', dg[perm_s, 1])
        mul!(yxi_edge, sh1d_deriv', dg[perm_s, 2])
        
        # Compute metrics in-place
        @. dsdxi = sqrt(xxi_edge^2 + yxi_edge^2)
        @. nl[:,1] = yxi_edge/dsdxi
        @. nl[:,2] = -xxi_edge/dsdxi
        @. cnl = c[1]*nl[:,1] + c[2]*nl[:,2]
        @. tauc = abs(cnl)
        @. tau = taud + tauc
        
        # Pre-compute quadrature weights scaled by edge length
        quad_weights = master.gw1d .* dsdxi
        
        for i in 1:nps  # Loop over nodes on this edge
            idof = i + (s-1)*nps
            
            # Create test function vector - reuse pre-allocated array
            mg = reshape(sh1d[i,:], :, 1)
            
            for s1 in 1:3  # Loop over edges for basis
                # Pre-compute edge offset for better cache locality
                edge_offset = (s1-1)*nps
                
                for j in 1:nps  # Loop over nodes on edge s1
                    jdof = j + edge_offset
                    
                    # Create unit vector - reuse pre-allocated array
                    fill!(nul, 0.0)
                    if s == s1
                        nul[j] = 1.0
                    end
    
                    # Transform to quadrature points - use mul! for better performance
                    mul!(nug, sh1d_t, nul)
                
                    # Extract values from precomputed solutions
                    mul!(ug, sh1d_t, view(um0, perm_s, jdof))
                    mul!(qgx, sh1d_t, view(qm0, perm_s, 1, jdof))
                    mul!(qgy, sh1d_t, view(qm0, perm_s, 2, jdof))
                
                    # Compute numerical flux in-place
                    @. qh = cnl * nug[:,1] + nl[:,1] * qgx[:,1] + nl[:,2] * qgy[:,1] + tau * (ug[:,1] - nug[:,1])
                    @. qhi = quad_weights * qh
   
                    # Update element matrix using efficient dot product
                    ae[idof, jdof] = -dot(mg, qhi)
                end
            end

            # Compute force vector contribution
            # Reuse pre-allocated arrays
            mul!(ug, sh1d_t, view(u0f, perm_s))
            mul!(qgx, sh1d_t, view(q0f, perm_s, 1))
            mul!(qgy, sh1d_t, view(q0f, perm_s, 2))
                
            # Numerical flux for source contribution - compute in-place
            @. qh = nl[:,1] * qgx[:,1] + nl[:,2] * qgy[:,1] + tau * ug[:,1]
            @. qhi = quad_weights * qh
        
            # Update force vector 
            fe[idof] = dot(mg, qhi)
        end
    end

    return ae, fe
end

"""
    hdg_solve(master, mesh, source, dbc, param)

Solves the convection-diffusion equation using the HDG method with a direct
(sparse LU) solve of the statically condensed trace system. Builds exactly
the same system as [`hdg_parsolve`](@ref) (element matrices via
`hdg_elemmats`, strong Dirichlet rows on boundary faces), so the two solvers
agree to solver — not just discretization — accuracy.

# Arguments
- `master`: Master element structure
- `mesh`: Mesh structure
- `source`: Source term function or nothing
- `dbc`: Dirichlet boundary condition data
- `param`: Dictionary with parameters `:kappa` (diffusivity), `:c` (convective
  velocity) and `:taud` (stabilization)

# Returns
- `uh (npl, 1, nt)`: Approximate scalar variable
- `qh (npl, 2, nt)`: Approximate flux
- `uhath (nps, nf)`: Approximate trace
"""
function hdg_solve(master, mesh, source, dbc, param)
    nps = mesh.porder + 1
    nt = size(mesh.t, 1)
    nf = size(mesh.f, 1)

    # Element matrices/vectors with strong Dirichlet rows already applied —
    # identical to the iterative path
    ae, fe = hdg_elemmats(master, mesh, source, dbc, param)

    elcon = mesh.elcon  # Element connectivity (maps local DOFs to global DOFs)

    # Assemble the global system from element contributions. Boundary faces
    # belong to a single element, so their identity rows/BC values pass
    # through assembly unchanged.
    ℍ = zeros(nf * nps, nf * nps)
    ℝ = zeros(nf * nps)
    for i in 1:nt
        global_inds = vec(elcon[:, :, i])
        ℍ[global_inds, global_inds] .+= ae[:, :, i]
        ℝ[global_inds] .+= fe[:, i]
    end

    # Solve the global system for the hybrid variable uhat
    uhath = reshape(sparse(ℍ) \ ℝ, nps, nf)

    # Local recovery step: reconstruct element-local solutions from uhat
    uh, qh = hdg_localrecovery(master, mesh, vec(uhath), source, param)

    return uh, qh, uhath
end