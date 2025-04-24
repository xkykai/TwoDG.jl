using LinearAlgebra
using SparseArrays

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
function localprob(dg, master, m, source, param)
    # Extract parameters
    kappa = param[:kappa]  # Diffusion coefficient
    c = param[:c]          # Convective velocity vector
    taud = kappa           # Diffusion stabilization parameter
    
    porder = master.porder
    nps = porder + 1
    ncol = size(m, 2)
    npl = size(dg, 1)
    
    perm = master.perm[:,:,1]  # Adjusted for 1-based indexing
    
    qmf = zeros(npl, 2, ncol)
    
    Fx = zeros(npl, ncol)
    Fy = zeros(npl, ncol)
    Fu = zeros(npl, ncol)
    
    # Volume integral
    shap = view(master.shap, :, 1, :)
    shapxi = view(master.shap, :, 2, :)
    shapet = view(master.shap, :, 3, :)
    
    # Pre-compute diagonal matrices
    gwgh_diag = Diagonal(master.gwgh)
    shapxig = shapxi * gwgh_diag  # Shape function derivatives weighted by quadrature weights
    shapetg = shapet * gwgh_diag  # Shape function derivatives weighted by quadrature weights
    
    # Compute Jacobian terms
    # These transform from reference to physical space
    xxi = shapxi' * dg[:,1]  # ∂x/∂ξ
    xet = shapet' * dg[:,1]  # ∂x/∂η
    yxi = shapxi' * dg[:,2]  # ∂y/∂ξ 
    yet = shapet' * dg[:,2]  # ∂y/∂η
    jac = xxi .* yet - xet .* yxi  # Determinant of the Jacobian matrix
    
    # Shape derivatives in physical space (using chain rule)
    shapx = shapxig * Diagonal(yet) - shapetg * Diagonal(yxi)  # ∂/∂x = (∂/∂ξ)(∂ξ/∂x) + (∂/∂η)(∂η/∂x)
    shapy = -shapxig * Diagonal(xet) + shapetg * Diagonal(xxi) # ∂/∂y = (∂/∂ξ)(∂ξ/∂y) + (∂/∂η)(∂η/∂y)
    
    # Mass and derivative matrices
    gwgh_jac = master.gwgh .* jac  # Quadrature weights scaled by Jacobian for physical integration
    M = (shap * Diagonal(gwgh_jac) * shap') / kappa  # Mass matrix for diffusion term (scaled by kappa^-1)
    Cx = shap * shapx'  # Coupling matrix for x-derivatives (∫ φ_i ∂φ_j/∂x)
    Cy = shap * shapy'  # Coupling matrix for y-derivatives (∫ φ_i ∂φ_j/∂y)
    
    # Convection operator: -c·∇u term from the PDE
    D = -c[1] .* Cx' - c[2] .* Cy'
    
    # Process source term if provided
    if source isa Function
        pg = shap' * dg
        src = source(pg)
        Fu = shap * Diagonal(gwgh_jac) * src
    end
    
    # Extract 1D shape functions
    sh1d = master.sh1d[:,1,:]
    sh1d_deriv = master.sh1d[:,2,:]
    
    # First loop: Compute stabilization terms
    # In HDG, stabilization plays a crucial role in coupling the element-local solutions
    for s in 1:3
        perm_s = @view perm[:, s]
        
        # Calculate normal vectors and Jacobian on the edge
        xxi_edge = sh1d_deriv' * view(dg, perm_s, 1)
        yxi_edge = sh1d_deriv' * view(dg, perm_s, 2)
        dsdxi = sqrt.(xxi_edge.^2 + yxi_edge.^2)  # Edge length scaling
        nl = hcat(yxi_edge./dsdxi, -xxi_edge./dsdxi)  # Outward normal vectors
        
        # Normal component of convection velocity
        cnl = c[1] * nl[:,1] + c[2] * nl[:,2]
        
        # Upwinding for convection and diffusion stabilization
        tauc = abs.(cnl)  # Upwind parameter for convection
        tau = taud .+ tauc  # Combined stabilization parameter
        
        # Add stabilization to diffusion matrix (penalizes jumps at interfaces)
        D[perm_s, perm_s] .+= sh1d * Diagonal(master.gw1d .* dsdxi .* tau) * sh1d'
    end
    
    # Second loop: Edge contributions to right-hand side
    # This implements the numerical flux that couples the element with neighbors through uhat
    for s in 1:3
        perm_s = view(perm, :, s)
        
        # Recompute normal vectors and Jacobian
        xxi_edge = sh1d_deriv' * view(dg, perm_s, 1)
        yxi_edge = sh1d_deriv' * view(dg, perm_s, 2)
        dsdxi = sqrt.(xxi_edge.^2 + yxi_edge.^2)
        nl = hcat(yxi_edge./dsdxi, -xxi_edge./dsdxi)
        
        cnl = c[1] * nl[:,1] + c[2] * nl[:,2]
        tauc = abs.(cnl)
        tau = taud .+ tauc
        
        # Pre-compute edge matrices for efficiency
        # These matrices represent the weak form of the boundary terms in the HDG formulation
        edge_matrix_x = sh1d * Diagonal(master.gw1d .* dsdxi .* nl[:,1]) * sh1d'  # x-component of flux
        edge_matrix_y = sh1d * Diagonal(master.gw1d .* dsdxi .* nl[:,2]) * sh1d'  # y-component of flux
        edge_matrix_u = sh1d * Diagonal(master.gw1d .* dsdxi .* (cnl .- tau)) * sh1d'  # convection-stabilization term
        
        for icol in 1:ncol
            ml = view(m, (s-1)*nps+1:s*nps, icol)  # Extract trace values for this edge
            
            # Apply boundary conditions through the hybrid variable m (uhat)
            view(Fx, perm_s, icol) .-= edge_matrix_x * ml  # Add x-flux contribution
            view(Fy, perm_s, icol) .-= edge_matrix_y * ml  # Add y-flux contribution 
            view(Fu, perm_s, icol) .-= edge_matrix_u * ml  # Add convection-stabilization contribution
        end
    end
    
    # Solve linear systems efficiently
    # First solve for auxiliary variables to compute the mixed formulation
    M1Fx = M \ Fx  # M^(-1) * Fx
    M1Fy = M \ Fy  # M^(-1) * Fy
    
    # Only compute M inverse once since it's used multiple times
    M1 = inv(M)
    
    # Compute final solutions
    # This is the "local solver" part of HDG - solving for u given uhat
    # system_matrix corresponds to the discretized convection-diffusion operator
    system_matrix = D + Cx * M1 * Cx' + Cy * M1 * Cy'
    umf = system_matrix \ (Fu - Cx * M1Fx - Cy * M1Fy)  # Solve for local solution u
    
    # Recover the flux q = -κ∇u
    qmf[:,1,:] = M1Fx + M \ (Cx' * umf)  # x-component of flux
    qmf[:,2,:] = M1Fy + M \ (Cy' * umf)  # y-component of flux
    
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
function elemmat_hdg(dg, master, source, param)
    nps = master.porder + 1

    kappa = param[:kappa]
    c = param[:c]
    taud = param[:taud]  # Stabilization parameter

    # Identity matrix for local problem
    mu = I(3*nps)  # Unit test functions for the hybrid variable uhat
    # Solve local problems with unit values of uhat to build the global system
    um0, qm0 = localprob(dg, master, mu, nothing, param)

    # Zero matrix for force vector computation
    m = zeros(3*nps, 1)  # Zero hybrid variable but with source term
    # Solve local problems with zero uhat but with source term
    u0f, q0f = localprob(dg, master, m, source, param)

    # Initialize element matrix and force vector
    ae = zeros(3*nps, 3*nps)  # Element stiffness matrix for the condensed system
    fe = zeros(3*nps)         # Element load vector for the condensed system
    
    perm = master.perm[:,:,1]  # Adjusted for 1-based indexing
    sh1d = master.sh1d[:,1,:]
    sh1d_deriv = master.sh1d[:,2,:]
    
    # Precompute transpose of shape functions
    sh1d_t = sh1d'
    
    for s in 1:3  # Loop over the 3 edges of the triangle
        perm_s = view(perm, :, s)
        
        # Calculate normal vectors and Jacobian
        xxi_edge = sh1d_deriv' * view(dg, perm_s, 1)
        yxi_edge = sh1d_deriv' * view(dg, perm_s, 2)
        dsdxi = sqrt.(xxi_edge.^2 + yxi_edge.^2)
        nl = hcat(yxi_edge./dsdxi, -xxi_edge./dsdxi)

        cnl = c[1]*nl[:,1] + c[2]*nl[:,2]
    
        tauc = abs.(cnl)
        tau = taud .+ tauc
    
        for i in 1:nps  # Loop over nodes on this edge
            idof = i + (s-1)*nps  # Global DOF index for this node
            
            # Get ith row of sh1d as a column vector
            mg = reshape(sh1d[i,:], :, 1)  # Test function for the conservation condition
            
            for s1 in 1:3  # Loop over all edges for the basis functions
                for j in 1:nps  # Loop over nodes on edge s1
                    jdof = j + (s1-1)*nps  # Global DOF index
                    
                    # Create unit vector for basis function
                    nul = zeros(nps, 1)
                    if s == s1
                        nul[j] = 1.0  # Set the basis function value to 1 at position j
                    end
    
                    nug = sh1d_t * nul  # Transform to quadrature points
                
                    # Extract values from precomputed solutions
                    ug = sh1d_t * view(um0, perm_s, jdof)       # u value on edge
                    qgx = sh1d_t * view(qm0, perm_s, 1, jdof)   # x-component of flux
                    qgy = sh1d_t * view(qm0, perm_s, 2, jdof)   # y-component of flux
                
                    # Compute numerical flux: F̂ = q·n + τ(u-û) + c·n·û
                    # This is the HDG numerical flux that ensures conservation
                    qh = cnl .* nug[:,1] + nl[:,1] .* qgx + nl[:,2] .* qgy + tau .* (ug .- nug[:,1])
                    qhi = master.gw1d .* dsdxi .* qh  # Scale by quadrature weights and edge length
   
                    # Update element matrix - this enforces the conservation condition
                    # ∫ μ·(q·n + τ(u-û)) ds = 0
                    ae[idof, jdof] = -dot(mg, qhi)
                end
            end

            # Compute force vector contribution from source term
            ug = sh1d_t * view(u0f, perm_s)
            qgx = sh1d_t * view(q0f, perm_s, 1)
            qgy = sh1d_t * view(q0f, perm_s, 2)
                
            # Numerical flux with just the source contribution
            qh = nl[:,1] .* qgx[:,1] + nl[:,2] .* qgy[:,1] + tau .* ug[:,1]
            qhi = master.gw1d .* dsdxi .* qh
        
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