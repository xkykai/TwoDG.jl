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
    ni = findfirst(x -> x < 0, mesh.f[:, 4]) - 1  # Julia uses 1-based indexing
    
    # Interior first
    Threads.@threads for i in 1:ni
        ipt = sum(mesh.f[i, 1:2])
        el = mesh.f[i, 3]
        er = mesh.f[i, 4]
        
        ipl = sum(mesh.t[el, :]) - ipt
        isl = findfirst(x -> x == ipl, mesh.t[el, :])
        if mesh.t2f[el, isl] > 0
            iol = 1  # Julia uses 1-based indexing
        else
            iol = 2  # Julia uses 1-based indexing
        end
        
        ipr = sum(mesh.t[er, :]) - ipt
        isr = findfirst(x -> x == ipr, mesh.t[er, :])
        if mesh.t2f[er, isr] > 0
            ior = 1  # Julia uses 1-based indexing
        else
            ior = 2  # Julia uses 1-based indexing
        end
        
        perml = perm[:, isl, iol]
        permr = perm[:, isr, ior]
        
        # INSERT CODE HERE .....
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

        û = ulg
        
        cntx = sh1d * Diagonal(dws .* nl[:, 1]) * û
        cnty = sh1d * Diagonal(dws .* nl[:, 2]) * û

        q[perml, 1, :, el] .+= cntx
        q[perml, 2, :, el] .+= cnty
        q[permr, 1, :, er] .-= cntx
        q[permr, 2, :, er] .-= cnty
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
        
        # INSERT CODE HERE .....
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
        
        ub = app.fvisub(ulg, nl, app.bcm[ib], app.bcs[app.bcm[ib], :], plg, app.arg, time)  # This line is incomplete in the original code

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
        # INSERT CODE HERE .....
        # Set up the coordinates and Jacobian for volume integration
        if mesh.tcurved[i]
            # For curved elements, compute Jacobian at each quadrature point
            xxi = @view(mesh.dgnodes[:, 1, i])' * shapxi
            xet = @view(mesh.dgnodes[:, 1, i])' * shapet
            yxi = @view(mesh.dgnodes[:, 2, i])' * shapxi
            yet = @view(mesh.dgnodes[:, 2, i])' * shapet
            jac = xxi .* yet - xet .* yxi
            shapx = shapxig * Diagonal(yet) - shapetg * Diagonal(yxi)
            shapy = -shapxig * Diagonal(xet) + shapetg * Diagonal(xxi)
            M = shap * Diagonal(master.gwgh .* jac) * shap'
        else
            # For straight elements, compute constant Jacobian
            xxi = mesh.p[mesh.t[i, 2], 1] - mesh.p[mesh.t[i, 1], 1]
            xet = mesh.p[mesh.t[i, 3], 1] - mesh.p[mesh.t[i, 1], 1]
            yxi = mesh.p[mesh.t[i, 2], 2] - mesh.p[mesh.t[i, 1], 2]
            yet = mesh.p[mesh.t[i, 3], 2] - mesh.p[mesh.t[i, 1], 2]
            jac = xxi * yet - xet * yxi
            shapx = shapxig * yet - shapetg * yxi
            shapy = -shapxig * xet + shapetg * xxi
            M = master.mass * jac
        end

        # Calculate the contribution to gradient from the element interior
        # Form the matrices for computing gradients in x and y directions
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
        q = getq(master, mesh, app, u, time)
    end

    r = zero(u)

    # Interfaces
    sh1d = master.sh1d[:, 1, :]# Julia uses 1-based indexing
    perm = master.perm
    ni = findfirst(x -> x < 0, mesh.f[:, 4]) - 1  # Julia uses 1-based indexing

    # Interior first
    Threads.@threads for i in 1:ni
        ipt = sum(mesh.f[i, 1:2])
        el = mesh.f[i, 3]
        er = mesh.f[i, 4]
        
        ipl = sum(mesh.t[el, :]) - ipt
        isl = findfirst(x -> x == ipl, mesh.t[el, :])
        if mesh.t2f[el, isl] > 0
            iol = 1  # Julia uses 1-based indexing
        else
            iol = 2  # Julia uses 1-based indexing
        end
        
        ipr = sum(mesh.t[er, :]) - ipt
        isr = findfirst(x -> x == ipr, mesh.t[er, :])
        if mesh.t2f[er, isr] > 0
            ior = 1  # Julia uses 1-based indexing
        else
            ior = 2  # Julia uses 1-based indexing
        end
        
        perml = perm[:, isl, iol]
        permr = perm[:, isr, ior]

        # Use conditional ? syntax for shorter code
        plg = app.pg ? sh1d' * @view(mesh.dgnodes[perml, :, el]) : []

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

        ur = @view(u[permr, :, er])
        urg = sh1d' * ur

        fng = app.finvi(ulg, urg, nl, plg, app.arg, time)

        if app.fvisv !== nothing
            qr = @view(q[permr, :, :, er])
            qrg = reshape(sh1d' * reshape(qr, (np1d, 2*nc)), (ng1d, 2, nc))
            fnvg = app.fvisi(ulg, urg, nothing, qrg, nl, plg, app.arg, time)
            fng = fng + fnvg
        end

        cnt = sh1d * Diagonal(dws) * fng

        ci = reshape(cnt, (np1d, nc))
        r[perml, :, el] = r[perml, :, el] - ci
        r[permr, :, er] = r[permr, :, er] + ci
    end

    # Now Boundary
    Threads.@threads for i in (ni+1):size(mesh.f, 1)
        ipt = sum(mesh.f[i, 1:2])
        el = mesh.f[i, 3]
        ib = -mesh.f[i, 4]  # Keep 0-based for the ib index since we're using it as array index

        ipl = sum(mesh.t[el, :]) - ipt
        isl = findfirst(x -> x == ipl, mesh.t[el, :])
        if mesh.t2f[el, isl] > 0
            iol = 1
        else
            iol = 2
        end

        perml = perm[:, isl, iol]

        plg = app.pg ? sh1d' * @view(mesh.dgnodes[perml, :, el]) : []

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
        
        fng = app.finvb(ulg, nl, app.bcm[ib], app.bcs[app.bcm[ib], :], plg, app.arg, time)

        if app.fvisv !== nothing
            ql = @view(q[perml, :, :, el])
            qlg = reshape(sh1d' * reshape(ql, (np1d, 2*nc)), (ng1d, 2, nc))
            fnvg = app.fvisb(ulg, qlg, nl, app.bcm[ib], app.bcs[app.bcm[ib], :], plg, app.arg, time)
            fng = fng + fnvg
        end
        
        cnt = sh1d * Diagonal(dws) * fng

        ci = reshape(cnt, (np1d, nc))
        r[perml, :, el] = r[perml, :, el] - ci
    end

    # Volume integral
    shap = master.shap[:, 1, :]
    shapxi = master.shap[:, 2, :]
    shapet = master.shap[:, 3, :]
    shapxig = shapxi * Diagonal(master.gwgh)
    shapetg = shapet * Diagonal(master.gwgh)

    Threads.@threads for i in 1:nt
        pg = app.pg ? shap' * @view(mesh.dgnodes[:, :, i]) : []

        if mesh.tcurved[i]
            xxi = @view(mesh.dgnodes[:, 1, i])' * shapxi
            xet = @view(mesh.dgnodes[:, 1, i])' * shapet
            yxi = @view(mesh.dgnodes[:, 2, i])' * shapxi
            yet = @view(mesh.dgnodes[:, 2, i])' * shapet
            jac = xxi .* yet - xet .* yxi
            shapx = shapxig * Diagonal(yet) - shapetg * Diagonal(yxi)
            shapy = -shapxig * Diagonal(xet) + shapetg * Diagonal(xxi)
            M = shap * Diagonal(master.gwgh .* jac) * shap'
        else
            xxi = mesh.p[mesh.t[i, 2], 1] - mesh.p[mesh.t[i, 1], 1]
            xet = mesh.p[mesh.t[i, 3], 1] - mesh.p[mesh.t[i, 1], 1]
            yxi = mesh.p[mesh.t[i, 2], 2] - mesh.p[mesh.t[i, 1], 2]
            yet = mesh.p[mesh.t[i, 3], 2] - mesh.p[mesh.t[i, 1], 2]
            jac = xxi * yet - xet * yxi
            shapx = shapxig * yet - shapetg * yxi
            shapy = -shapxig * xet + shapetg * xxi 
            M = master.mass * jac
        end

        ug = shap' * @view(u[:, :, i])

        if app.src !== nothing
            src = app.src(ug, [], pg, app.arg, time)
            r[:, :, i] = r[:, :, i] + shap * Diagonal(master.gwgh .* jac) * src
        end

        fgx, fgy = app.finvv(ug, pg, app.arg, time)

        if app.fvisv !== nothing
            qg = reshape(shap' * reshape(@view(q[:, :, :, i]), (npl, 2*nc)), (ng, 2, nc))
            fxvg, fyvg = app.fvisv(ug, qg, pg, app.arg, time)
            fgx = fgx + fxvg
            fgy = fgy + fyvg
        end

        r[:, :, i] = r[:, :, i] + shapx * fgx + shapy * fgy

        r[:, :, i] = M \ r[:, :, i]  # Julia's \ operator for solving linear systems
    end

    return r
end