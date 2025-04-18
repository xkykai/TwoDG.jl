using LinearAlgebra
using Statistics

"""
rinvexpl calculates the residual vector for explicit time stepping.

   master:       master structure
   mesh:         mesh structure
   app:          application structure
   u(npl,nc,nt): vector of unknowns
                 npl = size(mesh.plocal,1)
                 nc = app.nc (number of equations in system)
                 nt = size(mesh.t,1)
   time:         time
   r(npl,nc,nt): residual vector (=du/dt) (already divided by mass
                 matrix)
"""
function rinvexpl(master, mesh, app, u, time)
    nt   = size(mesh.t, 1)
    nf   = size(mesh.f, 1)
    nc   = app.nc
    npl  = size(mesh.dgnodes, 1)
    np1d = size(master.perm, 1)
    ng   = length(master.gwgh)
    ng1d = length(master.gw1d)

    r = zeros(size(u))

    # Interfaces
    sh1d = master.sh1d[:, 1, :]
    perm = master.perm
    ni = findfirst(i -> mesh.f[i, 4] < 0, 1:nf) - 1

    fcurved = mesh.fcurved
    tcurved = mesh.tcurved

    # Interior faces first
    Threads.@threads for i in 1:ni
        ipt = mesh.f[i, 1] + mesh.f[i, 2]
        el  = mesh.f[i, 3]
        er  = mesh.f[i, 4]
    
        ipl = sum(mesh.t[el, :]) - ipt
        isl = findfirst(x -> x == ipl, mesh.t[el, :])
        iol = mesh.t2f[el, isl] < 0 ? 2 : 1  # Adjusted from 0/1 to 1/2

        ipr = sum(mesh.t[er, :]) - ipt
        isr = findfirst(x -> x == ipr, mesh.t[er, :])
        ior = mesh.t2f[er, isr] < 0 ? 2 : 1  # Adjusted from 0/1 to 1/2
    
        perml = perm[:, isl, iol]
        permr = perm[:, isr, ior]

        curved_f = fcurved[i]

        if app.pg
            plg = sh1d' * mesh.dgnodes[perml, :, el]
        else
            plg = []
        end

        if !curved_f
            dx = mesh.p[mesh.f[i, 2], :] .- mesh.p[mesh.f[i, 1], :]
            dsdxi = sqrt(sum(dx .^ 2))
            nl = repeat([dx[2], -dx[1]] ./ dsdxi, 1, ng1d)'  # Similar to np.matlib.repmat
        else
            coords = mesh.dgnodes[perml, :, el]
            nl = zeros(ng1d, 2)
            dsdxi = zeros(ng1d)
            for j in axes(nl, 1)
                τ = master.sh1d[:, 2, j]' * coords
                τ_norm = sqrt(sum(τ.^2))
                τ ./= τ_norm

                nl[j, :] .= [τ[2], -τ[1]]
                dsdxi[j] = τ_norm
            end
        end

        dws = master.gw1d .* dsdxi
 
        ul = u[perml, :, el]
        ur = u[permr, :, er]
        ulg = sh1d' * ul
        urg = sh1d' * ur

        fng = app.finvi(ulg, urg, nl, plg, app.arg, time)
        cnt = sh1d * Diagonal(dws) * fng
   
        r[perml, :, el] .-= cnt
        r[permr, :, er] .+= cnt
    end

    # # Boundary faces
    Threads.@threads for i in ni+1:nf
        ipt = mesh.f[i, 1] + mesh.f[i, 2]
        el  = mesh.f[i, 3]
        ib  = -mesh.f[i, 4]  # This gives a 1-based index in Julia

        ipl = sum(mesh.t[el, :]) - ipt
        isl = findfirst(x -> x == ipl, mesh.t[el, :])
        iol = mesh.t2f[el, isl] < 0 ? 2 : 1  # Adjusted from 0/1 to 1/2

        perml = vec(perm[:, isl, iol])

        curved_f = fcurved[i]

        if app.pg
            plg = sh1d' * mesh.dgnodes[perml, :, el]
        else
            plg = []
        end

        if !curved_f
            dx = mesh.p[mesh.f[i, 2], :] .- mesh.p[mesh.f[i, 1], :]
            dsdxi = sqrt(sum(dx .^ 2))
            nl = repeat([dx[2], -dx[1]] ./ dsdxi, 1, ng1d)'  # Similar to np.matlib.repmat
        else
            coords = mesh.dgnodes[perml, :, el]
            nl = zeros(ng1d, 2)
            dsdxi = zeros(ng1d)
            for j in axes(nl, 1)
                τ = master.sh1d[:, 2, j]' * coords
                τ_norm = sqrt(sum(τ.^2))
                τ ./= τ_norm
                
                nl[j, :] .= [τ[2], -τ[1]]
                dsdxi[j] = τ_norm
            end
        end

        dws = master.gw1d .* dsdxi

        ul = u[perml, :, el]
        ulg = sh1d' * ul
        
        fng = app.finvb(ulg, nl, app.bcm[ib], app.bcs[app.bcm[ib], :], plg, app.arg, time)
        cnt = sh1d * Diagonal(dws) * fng
   
        r[perml, :, el] .-= cnt
    end

    # Volume integral
    shap = master.shap[:, 1, :]
    shapxi = master.shap[:, 2, :]
    shapet = master.shap[:, 3, :]
    shapxig = shapxi * Diagonal(master.gwgh)
    shapetg = shapet * Diagonal(master.gwgh)

    Threads.@threads for i in 1:nt
        if app.pg
            pg = shap' * mesh.dgnodes[:, :, i]
        else
            pg = []
        end

        curved_t = tcurved[i]

        if !curved_t
            xxi = mesh.p[mesh.t[i, 2], 1] - mesh.p[mesh.t[i, 1], 1]
            xet = mesh.p[mesh.t[i, 3], 1] - mesh.p[mesh.t[i, 1], 1]
            yxi = mesh.p[mesh.t[i, 2], 2] - mesh.p[mesh.t[i, 1], 2]
            yet = mesh.p[mesh.t[i, 3], 2] - mesh.p[mesh.t[i, 1], 2]
            detJ = xxi * yet - xet * yxi
            shapx =   shapxig * yet - shapetg * yxi
            shapy = - shapxig * xet + shapetg * xxi

            M = master.mass .* detJ
        else
            coords = mesh.dgnodes[:, :, i]
            detJ = zeros(ng)
            shapx = zeros(npl, ng)
            shapy = zeros(npl, ng)

            for j in eachindex(master.gwgh)
                J = master.shap[:, 2:3, j]' * coords
                invJ = inv(J)
                detJ[j] = det(J)
                shap∇ = invJ * master.shap[:, 2:3, j]'
                shapx[:, j] .= shap∇[1, :] .* master.gwgh[j] .* detJ[j]
                shapy[:, j] .= shap∇[2, :] .* master.gwgh[j] .* detJ[j]
            end

            M = shap * Diagonal(master.gwgh .* detJ) * shap'
        end

        ug = shap' * u[:, :, i]
   
        if app.src !== nothing
            src = app.src(ug, [], pg, app.arg, time)
            r[:, :, i] .+= shap * Diagonal(master.gwgh .* detJ) * src
        end
   
        fgx, fgy = app.finvv(ug, pg, app.arg, time)

        r[:, :, i] .+= shapx * fgx .+ shapy * fgy

        r[:, :, i] .= M \ r[:, :, i]  # More efficient than inv(M) * r
    end

    return r
end