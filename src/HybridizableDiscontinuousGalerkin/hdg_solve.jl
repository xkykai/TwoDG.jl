using LinearAlgebra

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
function localprob(dg::Matrix{T}, master, m::Matrix{T}, source, param::Dict) where T <: AbstractFloat
    # Extract parameters
    kappa = convert(T, param[:kappa])
    c = convert(Vector{T}, param[:c])
    taud = kappa
    
    porder = master.porder
    nps = porder + 1
    ncol = size(m, 2)
    npl = size(dg, 1)
    
    perm = master.perm[:,:,1]  # Adjusted for 1-based indexing
    
    qmf = zeros(T, npl, 2, ncol)
    
    Fx = zeros(T, npl, ncol)
    Fy = zeros(T, npl, ncol)
    Fu = zeros(T, npl, ncol)
    
    # Volume integral
    shap = view(master.shap, :, 1, :)
    shapxi = view(master.shap, :, 2, :)
    shapet = view(master.shap, :, 3, :)
    
    # Pre-compute diagonal matrices
    gwgh_diag = Diagonal(master.gwgh)
    shapxig = shapxi * gwgh_diag
    shapetg = shapet * gwgh_diag
    
    # Compute Jacobian terms
    xxi = dg[:,1] * shapxi
    xet = dg[:,1] * shapet
    yxi = dg[:,2] * shapxi
    yet = dg[:,2] * shapet
    jac = xxi .* yet - xet .* yxi
    
    # Shape derivatives
    shapx = shapxig * Diagonal(yet) - shapetg * Diagonal(yxi)
    shapy = -shapxig * Diagonal(xet) + shapetg * Diagonal(xxi)
    
    # Mass and derivative matrices
    gwgh_jac = master.gwgh .* jac
    M = (shap * Diagonal(gwgh_jac) * shap') / kappa
    Cx = shap * shapx'
    Cy = shap * shapy'
    
    D = -c[1] * Cx' - c[2] * Cy'
    
    # Process source term if provided
    if source !== nothing
        pg = shap' * dg
        src = source(pg)
        Fu = shap * Diagonal(gwgh_jac) * src
    end
    
    # Extract 1D shape functions
    sh1d = master.sh1d[:,1,:]
    sh1d_deriv = master.sh1d[:,2,:]
    
    # First loop: Compute stabilization terms
    for s in 1:3
        perm_s = view(perm, :, s)
        
        # Calculate normal vectors and Jacobian
        xxi_edge = sh1d_deriv' * view(dg, perm_s, 1)
        yxi_edge = sh1d_deriv' * view(dg, perm_s, 2)
        dsdxi = sqrt.(xxi_edge.^2 + yxi_edge.^2)
        nl = hcat(yxi_edge./dsdxi, -xxi_edge./dsdxi)
        
        cnl = c[1] * nl[:,1] + c[2] * nl[:,2]
        
        tauc = abs.(cnl)
        tau = taud .+ tauc
        
        # Add stabilization to diffusion matrix
        D[perm_s, perm_s] += sh1d * Diagonal(master.gw1d .* dsdxi .* tau) * sh1d'
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
function elemmat_hdg(dg::Matrix{T}, master, source, param::Dict) where T <: AbstractFloat
    nps = master.porder + 1

    kappa = convert(T, param[:kappa])
    c = convert(Vector{T}, param[:c])
    taud = kappa

    # Identity matrix for local problem
    mu = Matrix{T}(I, 3*nps, 3*nps)
    um0, qm0 = localprob(dg, master, mu, nothing, param)

    # Zero matrix for force vector computation
    m = zeros(T, 3*nps, 1)
    u0f, q0f = localprob(dg, master, m, source, param)

    # Initialize element matrix and force vector
    ae = zeros(T, 3*nps, 3*nps)
    fe = zeros(T, 3*nps)

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
                    nul = zeros(T, nps, 1)
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

using LinearAlgebra

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
function hdg_solve(master, mesh, source, dbc, param::Dict{Symbol,Any})
    nps = mesh.porder + 1
    npl = size(mesh.dgnodes, 1)
    nt = size(mesh.t, 1)
    nf = size(mesh.f, 1)

    # Initialize element matrices and force vectors
    ae = zeros(3*nps, 3*nps, nt)
    fe = zeros(3*nps, nt)

    # Pre-allocate for multi-threading if needed
    Threads.@threads for i in 1:nt
        # Get element nodes and compute element matrices
        element_nodes = view(mesh.dgnodes, :, :, i)
        ae[:,:,i], fe[:,i] = elemmat_hdg(element_nodes, master, source, param)
    end

    # YOUR CODE HERE ...
    
end