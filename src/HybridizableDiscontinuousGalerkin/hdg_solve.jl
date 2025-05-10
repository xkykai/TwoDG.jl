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

Solves the convection-diffusion equation using the HDG method.

# Arguments
- `master`: Master element structure
- `mesh`: Mesh structure
- `source`: Source term function or nothing
- `dbc`: Dirichlet boundary condition data
- `param`: Dictionary with parameters `:kappa` (diffusivity) and `:c` (convective velocity)

# Returns
- `uh`: Approximate scalar variable
- `qh`: Approximate flux
- `uhath`: Approximate trace
"""
function hdg_solve(master, mesh, source, dbc, param)
    nps = mesh.porder + 1
    npl = size(mesh.dgnodes, 1)
    nt = size(mesh.t, 1)
    nf = size(mesh.f, 1)

    # Initialize element matrices and force vectors
    ae = zeros(3*nps, 3*nps, nt)  # Element matrices for each triangle
    fe = zeros(3*nps, nt)         # Element force vectors for each triangle

    elcon = mesh.elcon  # Element connectivity information (maps local DOFs to global DOFs)

    # Initialize global system
    ℍ = zeros(nf * nps, nf * nps)  # Global system matrix (for uhat only, after static condensation)
    ℝ = zeros(nf * nps)            # Global right-hand side vector

    # Pre-allocate for multi-threading if needed
    Threads.@threads for i in 1:nt  # Compute local matrices in parallel
        # Get element nodes and compute element matrices
        element_nodes = view(mesh.dgnodes, :, :, i)
        a, f = elemmat_hdg(element_nodes, master, source, param)
        ae[:,:,i] .= a
        fe[:,i] .= f
    end

    # 1. First identify all boundary vertices
    # This section identifies nodes on the domain boundary for applying boundary conditions
    boundary_vertices = Dict()  # Maps vertex coordinates to global DOF indices
    global_boundary_nodes = Dict()  # Maps global DOF indices to boundary values

    is_boundary_nodes = zeros(Bool, nf * nps)  # Boolean flag for boundary nodes

    ni = findfirst(i -> mesh.f[i, 4] < 0, 1:nf)  # Find first boundary face
    
    # Scan boundary faces to find all boundary vertices
    for i in ni:nf  # ni is the start index of boundary faces
        it = mesh.f[i, 3]  # Element containing this face
        face_num = findfirst(x -> x == i, abs.(mesh.t2f[it, :]))
        orientation = mesh.t2f[it, face_num] > 0 ? 1 : 2
        perm = master.perm[:, face_num, orientation]
        face_coords = mesh.dgnodes[perm, :, it]
        
        # Add first and last nodes of face (endpoints) to boundary vertices
        boundary_vertices[face_coords[1, :]] = -mesh.f[i, 4]  # First node
        boundary_vertices[face_coords[end, :]] = -mesh.f[i, 4]  # Last node

        for j in 1:nps
            global_node_num = (i-1)*nps + j
            is_boundary_nodes[global_node_num] = true
            global_boundary_nodes[global_node_num] = dbc(face_coords[j, :])[1]
        end
    end
    
    # 2. Now scan interior faces to find vertices that lie on boundary but aren't part of boundary faces
    for i in 1:(ni-1)  # Interior faces
        it = mesh.f[i, 3]  # Left element
        
        # Get coordinates of the face endpoints
        face_num = findfirst(x -> x == i, abs.(mesh.t2f[it, :]))
        orientation = mesh.t2f[it, face_num] > 0 ? 1 : 2
        perm = master.perm[:, face_num, orientation]
        face_coords = mesh.dgnodes[perm, :, it]

        # Check if the face endpoints are in the boundary vertices
        if haskey(boundary_vertices, face_coords[1, :])
            global_node_num = (i-1)*nps + 1
            is_boundary_nodes[global_node_num] = true
            global_boundary_nodes[global_node_num] = dbc(face_coords[1, :])[1]
        end

        if haskey(boundary_vertices, face_coords[end, :])
            global_node_num = i*nps
            is_boundary_nodes[global_node_num] = true
            global_boundary_nodes[global_node_num] = dbc(face_coords[end, :])[1]
        end
    end
    
    # Assemble the global system from element contributions
    for i in 1:nt
        global_inds = vec(elcon[:, :, i])  # Maps element DOFs to global DOFs
        ℍ[global_inds, global_inds] .+= ae[:,:,i]  # Add element matrix to global matrix
        ℝ[global_inds] .+= fe[:, i]                # Add element vector to global vector
    end

    # Apply Dirichlet boundary conditions using the penalty method
    ℍ[is_boundary_nodes, :] .= 0  # Zero out rows for boundary nodes
    ℍ[:, is_boundary_nodes] .= 0  # Zero out columns for boundary nodes
    ℍ[is_boundary_nodes, is_boundary_nodes] .= I(sum(is_boundary_nodes))  # Set diagonal to identity
    
    for i in keys(global_boundary_nodes)
        ℝ[i] = global_boundary_nodes[i]  # Set RHS to boundary value
    end

    ℍ = sparse(ℍ)  # Convert to sparse matrix for efficient solving

    # Solve the global system for the hybrid variable uhat
    uhath = reshape(ℍ \ ℝ, nps, nf)

    # Initialize local solutions
    uh = zeros(npl, 1, nt)  # Scalar solution u
    qh = zeros(npl, 2, nt)  # Vector flux q

    # Local recovery step: reconstruct element-local solutions from uhat
    Threads.@threads for i in 1:nt
        element_nodes = view(mesh.dgnodes, :, :, i)
        t2f = mesh.t2f[i, :]  # Face indices for this element
        û = zeros(nps, 3)     # Hybrid variable values on element faces
        
        # Extract appropriate hybrid variable values for this element
        for (iface, face) in enumerate(t2f)
            if face > 0
                û[:, iface] .= uhath[:, face]  # Face orientation matches global orientation
            else
                û[:, iface] .= reverse(uhath[:, -face])  # Face orientation is reversed
            end
        end

        # Solve the local problem to recover u and q
        umf, qmf = localprob(element_nodes, master, vec(û), source, param)

        # Store the local solutions
        uh[:, 1, i] .= umf
        qh[:, 1, i] .= qmf[:, 1]  # x-component of flux
        qh[:, 2, i] .= qmf[:, 2]  # y-component of flux
    end

    return uh, qh, uhath
end