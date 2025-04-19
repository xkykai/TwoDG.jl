using LinearAlgebra

"""
    getq(master, mesh, app, u, time)

Calculate the gradient quantities q = ∇u for use in the LDG method.

# Arguments
- `master`: master structure
- `mesh`: mesh structure
- `app`: application structure
- `u`: vector of unknowns [npl,nc,nt]
       npl = size(mesh.plocal,1)
       nc = app.nc (number of equations in system)
       nt = size(mesh.t,1)
- `time`: time

# Returns
- `q`: gradient vector [npl,2,nc,nt]
"""
function getq(master, mesh, app, u, time)
    nt = size(mesh.t, 1)
    nc = app.nc
    npl = size(u, 1)
    ng1d = length(master.gw1d)
    
    q = zeros((npl, 2, nc, nt))
    
    # Interfaces
    sh1d = master.sh1d[:, 1, :] # Julia uses 1-based indexing
    perm = master.perm
    ni = findfirst(x -> x < 0, mesh.f[:, 4]) - 1  # First ni faces are interior, rest are boundary
    
    # Interior first
    Threads.@threads for i in 1:ni
        ipt = sum(mesh.f[i, 1:2])  # Sum of the two vertex indices forming this face
        el = mesh.f[i, 3]          # Left element index
        er = mesh.f[i, 4]          # Right element index
        
        # Find the local face index within the left element
        ipl = sum(mesh.t[el, :]) - ipt  # The vertex that's not part of this face
        isl = findfirst(x -> x == ipl, mesh.t[el, :])  # Local index of face in left element
        
        # Determine orientation of face relative to element (determines quadrature point ordering)
        if mesh.t2f[el, isl] > 0
            iol = 1  # Face orientation matches element orientation
        else
            iol = 2  # Face orientation is reversed relative to element
        end
        
        # Same procedure for right element
        ipr = sum(mesh.t[er, :]) - ipt
        isr = findfirst(x -> x == ipr, mesh.t[er, :])
        if mesh.t2f[er, isr] > 0
            ior = 1
        else
            ior = 2
        end
        
        perml = perm[:, isl, iol]
        permr = perm[:, isr, ior]
        
        if mesh.fcurved[i]
            xxi = master.sh1d[:, 2, :]' * @view(mesh.dgnodes[perml, 1, el])  # x-derivative along face
            yxi = master.sh1d[:, 2, :]' * @view(mesh.dgnodes[perml, 2, el])  # y-derivative along face
            dsdxi = sqrt.(xxi.^2 + yxi.^2)  # Jacobian of mapping from reference to physical face
            nl = hcat(yxi./dsdxi, -xxi./dsdxi)  # Outward normal vectors at quadrature points
            dws = master.gw1d .* dsdxi  # Quadrature weights adjusted for face geometry
        else
            dx = mesh.p[mesh.f[i, 2], :] - mesh.p[mesh.f[i, 1], :]  # Vector along face
            dsdxi = sqrt(dx[1]^2 + dx[2]^2)  # Face length
            nl = repeat([dx[2], -dx[1]]./dsdxi, 1, ng1d)'  # Outward normal (rotated tangent)
            dws = master.gw1d .* dsdxi  # Quadrature weights adjusted for face length
        end

        ul = @view(u[perml, :, el])
        ulg = sh1d' * ul

        û = ulg
        
        cntx = sh1d * Diagonal(dws .* nl[:, 1]) * û  # x-component of flux integral
        cnty = sh1d * Diagonal(dws .* nl[:, 2]) * û  # y-component of flux integral

        q[perml, 1, :, el] .+= cntx  # Add to left element's x-gradient
        q[perml, 2, :, el] .+= cnty  # Add to left element's y-gradient
        q[permr, 1, :, er] .-= cntx  # Subtract from right element's x-gradient
        q[permr, 2, :, er] .-= cnty  # Subtract from right element's y-gradient
    end

    # Now boundary
    Threads.@threads for i in (ni+1):size(mesh.f, 1)
        ipt = sum(mesh.f[i, 1:2])
        el = mesh.f[i, 3]
        ib = -mesh.f[i, 4]  # Keep 0-based for the ib index since we're using it as array index
        
        ipl = sum(mesh.t[el, :]) - ipt
        isl = findfirst(x -> x == ipl, mesh.t[el, :])
        if mesh.t2f[el, isl] > 0
            iol = 1  # Julia uses 1-based indexing
        else
            iol = 2  # Julia uses 1-based indexing
        end
        
        perml = perm[:, isl, iol]
        
        if app.pg
            plg = sh1d' * mesh.dgnodes[perml, :, el]
        else
            plg = []
        end

        # Get solution values on boundary element
        ul = u[perml, :, el]
        ulg = sh1d' * ul  # Interpolate solution to quadrature points

        if mesh.fcurved[i]
            xxi = master.sh1d[:, 2, :]' * @view(mesh.dgnodes[perml, 1, el])
            yxi = master.sh1d[:, 2, :]' * @view(mesh.dgnodes[perml, 2, el])
            dsdxi = sqrt.(xxi.^2 + yxi.^2)
            nl = hcat(yxi./dsdxi, -xxi./dsdxi)
            dws = master.gw1d .* dsdxi
        else
            dx = mesh.p[mesh.f[i, 2], :] - mesh.p[mesh.f[i, 1], :]
            dsdxi = sqrt(dx[1]^2 + dx[2]^2)
            nl = repeat([dx[2], -dx[1]]./dsdxi, 1, ng1d)'
            dws = master.gw1d .* dsdxi
        end
        
        ub = app.fvisub(ulg, nl, app.bcm[ib], app.bcs[app.bcm[ib], :], plg, app.arg, time)

        cntx = sh1d * Diagonal(master.gw1d .* dsdxi .* nl[:, 1]) * ub
        cnty = sh1d * Diagonal(master.gw1d .* dsdxi .* nl[:, 2]) * ub
        
        q[perml, 1, :, el] .+= cntx
        q[perml, 2, :, el] .+= cnty
    end
    
    # Volume integral
    shap = master.shap[:, 1, :]
    shapxi = master.shap[:, 2, :]
    shapet = master.shap[:, 3, :]
    shapxig = shapxi * Diagonal(master.gwgh)
    shapetg = shapet * Diagonal(master.gwgh)
    
    Threads.@threads for i in 1:nt
        curved_t = mesh.tcurved[i]  # Whether this element has curved edges
        ng   = length(master.gwgh)  # Number of Gaussian quadrature points

        if !curved_t
            xxi = mesh.p[mesh.t[i, 2], 1] - mesh.p[mesh.t[i, 1], 1]  # x-coordinate of first edge vector
            xet = mesh.p[mesh.t[i, 3], 1] - mesh.p[mesh.t[i, 1], 1]  # x-coordinate of second edge vector
            yxi = mesh.p[mesh.t[i, 2], 2] - mesh.p[mesh.t[i, 1], 2]  # y-coordinate of first edge vector
            yet = mesh.p[mesh.t[i, 3], 2] - mesh.p[mesh.t[i, 1], 2]  # y-coordinate of second edge vector
            detJ = xxi * yet - xet * yxi  # Determinant of Jacobian matrix (twice the element area)
            
            shapx =   shapxig * yet - shapetg * yxi  # x-derivatives in physical space
            shapy = - shapxig * xet + shapetg * xxi  # y-derivatives in physical space

            M = master.mass .* detJ  # Mass matrix scaled by element area
        else
            coords = mesh.dgnodes[:, :, i]  # Nodal coordinates for this element
            detJ = zeros(ng)
            shapx = zeros(npl, ng)
            shapy = zeros(npl, ng)

            for j in eachindex(master.gwgh)
                J = master.shap[:, 2:3, j]' * coords  # Jacobian matrix at this quadrature point
                invJ = inv(J)  # Inverse Jacobian
                detJ[j] = det(J)  # Determinant of Jacobian (local scaling factor)
                
                shap∇ = invJ * master.shap[:, 2:3, j]'
                shapx[:, j] .= shap∇[1, :] .* master.gwgh[j] .* detJ[j]  # x-derivatives weighted
                shapy[:, j] .= shap∇[2, :] .* master.gwgh[j] .* detJ[j]  # y-derivatives weighted
            end

            M = shap * Diagonal(master.gwgh .* detJ) * shap'
        end

        Cx = (shapx * shap')'
        Cy = (shapy * shap')'
        
        q[:, 1, :, i] .-= Cx' * @view(u[:, :, i])
        q[:, 2, :, i] .-= Cy' * @view(u[:, :, i])
        
        q_reshaped = reshape(q[:, :, :, i], (npl, 2*nc))
        q[:, :, :, i] = reshape(M \ q_reshaped, (npl, 2, nc))
    end
    
    return q
end

"""
    rldgexpl(master, mesh, app, u, time)

Calculate the residual vector for explicit time stepping using the LDG method.

# Arguments
- `master`: master structure
- `mesh`: mesh structure
- `app`: application structure
- `u`: vector of unknowns [npl,nc,nt]
       npl = size(mesh.plocal,1)
       nc = app.nc (number of equations in system)
       nt = size(mesh.t,1)
- `time`: time

# Returns
- `r`: residual vector (=du/dt) [npl,nc,nt]
       (already divided by mass matrix)
"""
function rldgexpl(master, mesh, app, u, time)
    nt = size(mesh.t, 1)
    nc = app.nc
    npl = size(u, 1)
    np1d = size(master.perm, 1)
    ng = length(master.gwgh)
    ng1d = length(master.gw1d)

    q = nothing
    if app.fvisv !== nothing
        # Calculate gradient vector using LDG method if viscous terms are present
        q = getq(master, mesh, app, u, time)
    end

    r = zero(u)

    # Interfaces
    sh1d = master.sh1d[:, 1, :]
    perm = master.perm
    # Find index where boundary faces start (mesh.f[:, 4] < 0 indicates boundary face)
    ni = findfirst(x -> x < 0, mesh.f[:, 4]) - 1

    # Process interior faces first
    Threads.@threads for i in 1:ni
        ipt = sum(mesh.f[i, 1:2])  # Sum of vertex indices forming this face
        el = mesh.f[i, 3]          # Left element index
        er = mesh.f[i, 4]          # Right element index
        
        # Determine which local face within the left element corresponds to this global face
        ipl = sum(mesh.t[el, :]) - ipt  # Find the vertex not on this face
        isl = findfirst(x -> x == ipl, mesh.t[el, :])  # Local face index opposite to this vertex
        
        # Determine face orientation relative to element (affects quadrature point ordering)
        if mesh.t2f[el, isl] > 0
            iol = 1  # Face orientation matches element orientation
        else
            iol = 2  # Face orientation is reversed
        end
        
        # Same procedure for right element
        ipr = sum(mesh.t[er, :]) - ipt
        isr = findfirst(x -> x == ipr, mesh.t[er, :])
        if mesh.t2f[er, isr] > 0
            ior = 1
        else
            ior = 2
        end
        
        # Get permutation indices to map from element local DOFs to face local DOFs
        perml = perm[:, isl, iol]
        permr = perm[:, isr, ior]

        # Get physical coordinates if needed (for nonlinear problems)
        plg = app.pg ? sh1d' * @view(mesh.dgnodes[perml, :, el]) : []

        # Calculate geometric information for this face
        if mesh.fcurved[i]
            # For curved faces, calculate Jacobian and normal vectors at each quadrature point
            xxi = master.sh1d[:, 2, :]' * @view(mesh.dgnodes[perml, 1, el])
            yxi = master.sh1d[:, 2, :]' * @view(mesh.dgnodes[perml, 2, el])
            dsdxi = sqrt.(xxi.^2 + yxi.^2)  # Length element (Jacobian)
            nl = hcat(yxi./dsdxi, -xxi./dsdxi)  # Outward normal vector (normalized)
            dws = master.gw1d .* dsdxi  # Scaled quadrature weights
        else
            # For straight faces, normal and Jacobian are constant
            dx = mesh.p[mesh.f[i, 2], :] - mesh.p[mesh.f[i, 1], :]
            dsdxi = sqrt(dx[1]^2 + dx[2]^2)  # Face length
            # Normal vector = rotated tangent vector (counterclockwise 90°)
            nl = repeat([dx[2], -dx[1]]./dsdxi, 1, ng1d)'
            dws = master.gw1d .* dsdxi
        end

        # Extract solution values on both sides of the face
        ul = @view(u[perml, :, el])
        ulg = sh1d' * ul  # Interpolate to face quadrature points

        ur = @view(u[permr, :, er])
        urg = sh1d' * ur  # Interpolate to face quadrature points

        # Calculate inviscid numerical flux at face
        fng = app.finvi(ulg, urg, nl, plg, app.arg, time)

        if app.fvisv !== nothing
            # Add viscous numerical flux if viscous terms are present
            qr = @view(q[permr, :, :, er])
            # Reshape to get gradients at face quadrature points
            qrg = reshape(sh1d' * reshape(qr, (np1d, 2*nc)), (ng1d, 2, nc))
            # Compute viscous interface flux (note: left element gradient not needed, hence nothing)
            fnvg = app.fvisi(ulg, urg, nothing, qrg, nl, plg, app.arg, time)
            fng = fng + fnvg
        end

        # Apply numerical flux to both elements (with opposite signs)
        cnt = sh1d * Diagonal(dws) * fng  # Integrate flux along face

        ci = reshape(cnt, (np1d, nc))
        r[perml, :, el] = r[perml, :, el] - ci  # Subtract from left element (outflow)
        r[permr, :, er] = r[permr, :, er] + ci  # Add to right element (inflow)
    end

    # Process boundary faces
    Threads.@threads for i in (ni+1):size(mesh.f, 1)
        ipt = sum(mesh.f[i, 1:2])
        el = mesh.f[i, 3]
        ib = -mesh.f[i, 4]  # Boundary type index (stored as negative number)

        # Find local face index in the element
        ipl = sum(mesh.t[el, :]) - ipt
        isl = findfirst(x -> x == ipl, mesh.t[el, :])
        if mesh.t2f[el, isl] > 0
            iol = 1
        else
            iol = 2
        end

        perml = perm[:, isl, iol]

        # Physical coordinates for nonlinear problems
        plg = app.pg ? sh1d' * @view(mesh.dgnodes[perml, :, el]) : []

        # Calculate face geometry information (similar to interior faces)
        if mesh.fcurved[i]
            xxi = master.sh1d[:, 2, :]' * @view(mesh.dgnodes[perml, 1, el])
            yxi = master.sh1d[:, 2, :]' * @view(mesh.dgnodes[perml, 2, el])
            dsdxi = sqrt.(xxi.^2 + yxi.^2)
            nl = hcat(yxi./dsdxi, -xxi./dsdxi)
            dws = master.gw1d .* dsdxi
        else
            dx = mesh.p[mesh.f[i, 2], :] - mesh.p[mesh.f[i, 1], :]
            dsdxi = sqrt(dx[1]^2 + dx[2]^2)
            nl = repeat([dx[2], -dx[1]]./dsdxi, 1, ng1d)'
            dws = master.gw1d .* dsdxi
        end

        ul = @view(u[perml, :, el])
        ulg = sh1d' * ul
        
        # Calculate boundary inviscid flux based on boundary type (bcm[ib])
        fng = app.finvb(ulg, nl, app.bcm[ib], app.bcs[app.bcm[ib], :], plg, app.arg, time)

        if app.fvisv !== nothing
            # Add viscous boundary flux if viscous terms are present
            ql = @view(q[perml, :, :, el])
            qlg = reshape(sh1d' * reshape(ql, (np1d, 2*nc)), (ng1d, 2, nc))
            fnvg = app.fvisb(ulg, qlg, nl, app.bcm[ib], app.bcs[app.bcm[ib], :], plg, app.arg, time)
            fng = fng + fnvg
        end
        
        # Apply boundary flux to element
        cnt = sh1d * Diagonal(dws) * fng
        ci = reshape(cnt, (np1d, nc))
        r[perml, :, el] = r[perml, :, el] - ci
    end

    # Volume integral (element contributions)
    shap = master.shap[:, 1, :]     # Shape functions
    shapxi = master.shap[:, 2, :]   # Shape function derivatives in ξ direction
    shapet = master.shap[:, 3, :]   # Shape function derivatives in η direction
    shapxig = shapxi * Diagonal(master.gwgh)  # Weighted shape function derivatives
    shapetg = shapet * Diagonal(master.gwgh)

    Threads.@threads for i in 1:nt
        # Get physical coordinates for quadrature points if needed
        pg = app.pg ? shap' * @view(mesh.dgnodes[:, :, i]) : []

        curved_t = mesh.tcurved[i]
        ng = length(master.gwgh)
        if !curved_t
            # For straight-sided elements, geometric terms are constant
            # Calculate edge vectors of the element
            xxi = mesh.p[mesh.t[i, 2], 1] - mesh.p[mesh.t[i, 1], 1]  # ∂x/∂ξ component
            xet = mesh.p[mesh.t[i, 3], 1] - mesh.p[mesh.t[i, 1], 1]  # ∂x/∂η component
            yxi = mesh.p[mesh.t[i, 2], 2] - mesh.p[mesh.t[i, 1], 2]  # ∂y/∂ξ component
            yet = mesh.p[mesh.t[i, 3], 2] - mesh.p[mesh.t[i, 1], 2]  # ∂y/∂η component
            
            # Determinant of Jacobian matrix (twice the element area)
            detJ = xxi * yet - xet * yxi
            
            # Convert reference derivatives to physical derivatives using chain rule
            # These include quadrature weights and Jacobian for integration
            shapx =   shapxig * yet - shapetg * yxi  # ∂/∂x = ∂/∂ξ * ∂ξ/∂x + ∂/∂η * ∂η/∂x
            shapy = - shapxig * xet + shapetg * xxi  # ∂/∂y = ∂/∂ξ * ∂ξ/∂y + ∂/∂η * ∂η/∂y

            # Mass matrix scaled by element area
            M = master.mass .* detJ
        else
            # For curved elements, geometric terms vary at each quadrature point
            coords = mesh.dgnodes[:, :, i]
            detJ = zeros(ng)
            shapx = zeros(npl, ng)
            shapy = zeros(npl, ng)

            for j in eachindex(master.gwgh)
                # Calculate Jacobian matrix at this quadrature point
                J = master.shap[:, 2:3, j]' * coords
                invJ = inv(J)  # Inverse Jacobian needed for derivatives
                detJ[j] = det(J)
                
                # Transform reference derivatives to physical derivatives
                shap∇ = invJ * master.shap[:, 2:3, j]'
                # Store weighted derivatives for integration
                shapx[:, j] .= shap∇[1, :] .* master.gwgh[j] .* detJ[j]  # x-derivatives
                shapy[:, j] .= shap∇[2, :] .* master.gwgh[j] .* detJ[j]  # y-derivatives
            end

            # Assemble mass matrix with varying Jacobian
            M = shap * Diagonal(master.gwgh .* detJ) * shap'
        end

        # Get solution at quadrature points
        ug = shap' * @view(u[:, :, i])

        # Add source term contribution if present
        if app.src !== nothing
            src = app.src(ug, [], pg, app.arg, time)
            r[:, :, i] = r[:, :, i] + shap * Diagonal(master.gwgh .* detJ) * src
        end

        # Calculate inviscid volume flux
        fgx, fgy = app.finvv(ug, pg, app.arg, time)

        # Add viscous volume flux if present
        if app.fvisv !== nothing
            # Transform gradient vector to quadrature points
            qg = reshape(shap' * reshape(@view(q[:, :, :, i]), (npl, 2*nc)), (ng, 2, nc))
            fxvg, fyvg = app.fvisv(ug, qg, pg, app.arg, time)
            fgx = fgx + fxvg
            fgy = fgy + fyvg
        end

        # Add flux divergence to residual (∇·f term)
        r[:, :, i] = r[:, :, i] + shapx * fgx + shapy * fgy

        # Apply inverse mass matrix to get final residual
        r[:, :, i] = M \ r[:, :, i]
    end

    return r
end