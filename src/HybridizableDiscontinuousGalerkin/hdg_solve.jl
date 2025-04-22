using LinearAlgebra
using SparseArrays

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
    kappa = param[:kappa]
    c = param[:c]
    taud = kappa
    
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
    shapxig = shapxi * gwgh_diag
    shapetg = shapet * gwgh_diag
    
    # Compute Jacobian terms
    xxi = shapxi' * dg[:,1]
    xet = shapet' * dg[:,1]
    yxi = shapxi' * dg[:,2]
    yet = shapet' * dg[:,2]
    jac = xxi .* yet - xet .* yxi
    
    # Shape derivatives
    shapx = shapxig * Diagonal(yet) - shapetg * Diagonal(yxi)
    shapy = -shapxig * Diagonal(xet) + shapetg * Diagonal(xxi)
    
    # Mass and derivative matrices
    gwgh_jac = master.gwgh .* jac
    M = (shap * Diagonal(gwgh_jac) * shap') / kappa
    Cx = shap * shapx'
    Cy = shap * shapy'
    
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
    for s in 1:3
        perm_s = @view perm[:, s]
        
        # Calculate normal vectors and Jacobian
        xxi_edge = sh1d_deriv' * view(dg, perm_s, 1)
        yxi_edge = sh1d_deriv' * view(dg, perm_s, 2)
        dsdxi = sqrt.(xxi_edge.^2 + yxi_edge.^2)
        nl = hcat(yxi_edge./dsdxi, -xxi_edge./dsdxi)
        
        cnl = c[1] * nl[:,1] + c[2] * nl[:,2]
        
        tauc = abs.(cnl)
        tau = taud .+ tauc
        
        # Add stabilization to diffusion matrix
        D[perm_s, perm_s] .+= sh1d * Diagonal(master.gw1d .* dsdxi .* tau) * sh1d'
    end
    
    # Second loop: Edge contributions to right-hand side
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
        edge_matrix_x = sh1d * Diagonal(master.gw1d .* dsdxi .* nl[:,1]) * sh1d'
        edge_matrix_y = sh1d * Diagonal(master.gw1d .* dsdxi .* nl[:,2]) * sh1d'
        edge_matrix_u = sh1d * Diagonal(master.gw1d .* dsdxi .* (cnl .- tau)) * sh1d'
        
        for icol in 1:ncol
            ml = view(m, (s-1)*nps+1:s*nps, icol)
            
            view(Fx, perm_s, icol) .-= edge_matrix_x * ml
            view(Fy, perm_s, icol) .-= edge_matrix_y * ml
            view(Fu, perm_s, icol) .-= edge_matrix_u * ml
        end
    end
    
    # Solve linear systems efficiently
    M1Fx = M \ Fx
    M1Fy = M \ Fy
    
    # Only compute M inverse once since it's used multiple times
    M1 = inv(M)
    
    # Compute final solutions
    system_matrix = D + Cx * M1 * Cx' + Cy * M1 * Cy'
    umf = system_matrix \ (Fu - Cx * M1Fx - Cy * M1Fy)
    
    qmf[:,1,:] = M1Fx + M \ (Cx' * umf)
    qmf[:,2,:] = M1Fy + M \ (Cy' * umf)
    
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
    taud = kappa

    # Identity matrix for local problem
    mu = I(3*nps)
    um0, qm0 = localprob(dg, master, mu, nothing, param)

    # Zero matrix for force vector computation
    m = zeros(3*nps, 1)
    u0f, q0f = localprob(dg, master, m, source, param)

    # Initialize element matrix and force vector
    ae = zeros(3*nps, 3*nps)
    fe = zeros(3*nps)

    perm = master.perm[:,:,1]  # Adjusted for 1-based indexing
    sh1d = master.sh1d[:,1,:]
    sh1d_deriv = master.sh1d[:,2,:]
    
    # Precompute transpose of shape functions
    sh1d_t = sh1d'
    
    for s in 1:3
        perm_s = view(perm, :, s)
        
        # Calculate normal vectors and Jacobian
        xxi_edge = sh1d_deriv' * view(dg, perm_s, 1)
        yxi_edge = sh1d_deriv' * view(dg, perm_s, 2)
        dsdxi = sqrt.(xxi_edge.^2 + yxi_edge.^2)
        nl = hcat(yxi_edge./dsdxi, -xxi_edge./dsdxi)

        cnl = c[1]*nl[:,1] + c[2]*nl[:,2]
    
        tauc = abs.(cnl)
        tau = taud .+ tauc
    
        for i in 1:nps
            idof = i + (s-1)*nps
            
            # Get ith row of sh1d as a column vector
            mg = reshape(sh1d[i,:], :, 1)
            
            for s1 in 1:3
                for j in 1:nps
                    jdof = j + (s1-1)*nps
                    
                    # Create unit vector for basis function
                    nul = zeros(nps, 1)
                    if s == s1
                        nul[j] = 1.0
                    end
    
                    nug = sh1d_t * nul
                
                    # Extract values from precomputed solutions
                    ug = sh1d_t * view(um0, perm_s, jdof)
                    qgx = sh1d_t * view(qm0, perm_s, 1, jdof) 
                    qgy = sh1d_t * view(qm0, perm_s, 2, jdof)
                
                    # Compute numerical flux
                    qh = cnl .* nug[:,1] + nl[:,1] .* qgx + nl[:,2] .* qgy + tau .* (ug .- nug[:,1])
                    qhi = master.gw1d .* dsdxi .* qh
   
                    # Update element matrix
                    ae[idof, jdof] = -dot(mg, qhi)
                end
            end

            # Compute force vector contribution
            ug = sh1d_t * view(u0f, perm_s)
            qgx = sh1d_t * view(q0f, perm_s, 1)
            qgy = sh1d_t * view(q0f, perm_s, 2)
                
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
    ae = zeros(3*nps, 3*nps, nt)
    fe = zeros(3*nps, nt)

    idm = mesh.idm

    ℍ = zeros(nf * nps, nf * nps)
    ℝ = zeros(nf * nps)

    # Pre-allocate for multi-threading if needed
    for i in 1:nt
        # Get element nodes and compute element matrices
        element_nodes = view(mesh.dgnodes, :, :, i)
        a, f = elemmat_hdg(element_nodes, master, source, param)
        ae[:,:,i] .= a
        fe[:,i] .= f
    end

    # 1. First identify all boundary vertices
    boundary_vertices = Dict()  # Maps vertex coordinates to global DOF indices
    global_boundary_nodes = Dict()

    is_boundary_nodes = zeros(Bool, nf * nps)

    ni = findfirst(i -> mesh.f[i, 4] < 0, 1:nf)

    # Scan boundary faces to find all boundary vertices
    for i in ni:nf  # ni is the start index of boundary faces
        it = mesh.f[i, 3]  # Element containing this face
        face_num = findfirst(x -> x == i, abs.(mesh.t2f[it, :]))
        orientation = mesh.t2f[it, face_num] > 0 ? 1 : 2
        perm = master.perm[:, face_num, orientation]
        face_coords = mesh.dgnodes[perm, :, it]
        
        # Add first and last nodes of face (endpoints) to boundary vertices
        boundary_vertices[face_coords[1, :]] = -mesh.f[i, 4]  # First node
        boundary_vertices[face_coords[end, :]] = -mesh.f[i, 4]        # Last node

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
    
    for i in 1:nt
        global_inds = vec(idm[:, :, i])
        ℍ[global_inds, global_inds] .+= ae[:,:,i]
        ℝ[global_inds] .+= fe[:, i]
    end

    ℍ[is_boundary_nodes, :] .= 0  # Set rows corresponding to boundary nodes to zero
    ℍ[:, is_boundary_nodes] .= 0  # Set columns corresponding to boundary nodes to zero
    ℍ[is_boundary_nodes, is_boundary_nodes] .= I(sum(is_boundary_nodes))  # Set diagonal to identity for boundary nodes

    for i in keys(global_boundary_nodes)
        ℝ[i] = global_boundary_nodes[i]  # Set the right-hand side for boundary nodes
    end

    uhath = reshape(ℍ \ ℝ, nps, nf)

    uh = zeros(npl, 1, nt)
    qh = zeros(npl, 2, nt)

    for i in 1:nt
        element_nodes = view(mesh.dgnodes, :, :, i)
        t2f = mesh.t2f[i, :]
        û = zeros(nps, 3)
        for (iface, face) in enumerate(t2f)
            if face > 0
                û[:, iface] .= uhath[:, face]
            else
                û[:, iface] .= reverse(uhath[:, -face])
            end
        end

        umf, qmf = localprob(element_nodes, master, vec(û), source, param)

        uh[:, 1, i] .= umf
        qh[:, 1, i] .= qmf[:, 1]
        qh[:, 2, i] .= qmf[:, 2]
    end

    return uh, qh, uhath

end