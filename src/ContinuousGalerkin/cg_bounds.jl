using LinearAlgebra
using SparseArrays
using TwoDG.Masters: shape1d, shape2d

"""
Computes the gradient of a scalar field on a mesh.

Parameters:
- master: master structure
- mesh: mesh structure
- uh: approximate scalar variable with local numbering, size (npl, nt)

Returns:
- guh: solution gradient, size (npl, 2, nt)
"""
function grad_u(master, mesh, uh)
    shap = shape2d(master.porder, master.plocal, master.plocal[:, 2:3])
    
    shapxi = shap[:, 2, :]
    shapet = shap[:, 3, :]
    
    npl, ne = size(uh)
    guh = zeros(npl, 2, ne)
    
    for i in 1:ne
        xxi = shapxi' * mesh.dgnodes[:, 1, i]
        xet = shapet' * mesh.dgnodes[:, 1, i]
        yxi = shapxi' * mesh.dgnodes[:, 2, i]
        yet = shapet' * mesh.dgnodes[:, 2, i]
        
        jac = xxi .* yet - xet .* yxi
        
        shapx = shapxi * Diagonal(yet ./ jac) - shapet * Diagonal(yxi ./ jac)
        shapy = -shapxi * Diagonal(xet ./ jac) + shapet * Diagonal(xxi ./ jac)
        
        guh[:, 1, i] = shapx' * uh[:, i]
        guh[:, 2, i] = shapy' * uh[:, i]
    end
    
    return guh
end

"""
Computes the normal flux on each edge ∇u · n by averaging the gradients 
from the two neighboring elements and then correcting them to ensure that 
the integral of qn along the element boundary is equal to the integral of f on the volume.

Parameters:
- master: master structure
- mesh: mesh structure
- guh: solution gradient, size (npl, 2, nt)
- forcing: forcing function

Returns:
- qn: equilibrated normal fluxes to the face. The direction is pointing into element mesh.f(i,3)
- qn0: non-equilibrated (averaged) normal fluxes to the face
"""
function equilibrate(master, mesh, guh, forcing)
    nt = size(mesh.t, 1)
    nf = size(mesh.f, 1)
    npl1d = size(master.ploc1d, 1)
    
    shap = master.shap[:, 1, :]
    
    M = zeros(npl1d, npl1d, nf)
    rhs = zeros(npl1d, nf)
    
    qn0 = zeros(npl1d, nf)
    sh1d = master.sh1d[:, 1, :]
    perm = master.perm
    
    # Loop over faces: average the flux from left/right elements
    for i in 1:nf
        ipt = sum(mesh.f[i, 1:2])
        el = mesh.f[i, 3]

        # find the local node in el that is opposite to this face
        ipl = sum(mesh.t[el, :]) - ipt
        isl = findfirst(x -> x == ipl, mesh.t[el, :])
        
        if mesh.t2f[el, isl] < 0
            iol = 2
        else
            iol = 1
        end
        
        # Coordinates of 1D quadrature points along the face in element el
        xxi = transpose(master.sh1d[:, 2, :]) * mesh.dgnodes[perm[:, isl, iol], 1, el]
        yxi = transpose(master.sh1d[:, 2, :]) * mesh.dgnodes[perm[:, isl, iol], 2, el]
        dsdxi = sqrt.(xxi.^2 .+ yxi.^2)
        
        # Normal vector
        nl = hcat(yxi ./ dsdxi, -xxi ./ dsdxi)
        dws = master.gw1d .* dsdxi
        
        # Gradient from "left" side
        gulx = guh[perm[:, isl, iol], 1, el]  # Python: guh[perm[:, isl, iol], 0, el]
        guly = guh[perm[:, isl, iol], 2, el]  # Python: guh[perm[:, isl, iol], 1, el]
        
        # Dot(gradient, normal) using shape function weighting
        ulgn = (nl[:, 1] .* (transpose(sh1d) * gulx) .+
                nl[:, 2] .* (transpose(sh1d) * guly))
        
        # Possibly an element "right" of this face
        er = mesh.f[i, 4]
        if er > 0
            # find local node in er that is opposite to the face
            ipr = sum(mesh.t[er, :]) - ipt
            isr = findfirst(x -> x == ipr, mesh.t[er, :])
            if mesh.t2f[er, isr] < 0
                ior = 2
            else
                ior = 1
            end
            
            # Right side gradient
            gurx = guh[perm[:, isr, ior], 1, er]
            gury = guh[perm[:, isr, ior], 2, er]
            urgn = (nl[:, 1] .* (transpose(sh1d) * gurx) .+
                    nl[:, 2] .* (transpose(sh1d) * gury))
            
            # Average left/right
            ulgn = 0.5 .* (ulgn .+ urgn)
        end
        
        Sd = Diagonal(dws)
        
        M[:, :, i] = sh1d * Sd * transpose(sh1d)
        rhs[:, i] = sh1d * Sd * ulgn
        qn0[:, i] = M[:, :, i] \ (sh1d * Sd * ulgn)
    end
    
    rs = zeros(nt)
    for i in 1:nt
        xxi = transpose(master.shap[:, 2, :]) * mesh.dgnodes[:, 1, i]
        xet = transpose(master.shap[:, 3, :]) * mesh.dgnodes[:, 1, i]
        yxi = transpose(master.shap[:, 2, :]) * mesh.dgnodes[:, 2, i]
        yet = transpose(master.shap[:, 3, :]) * mesh.dgnodes[:, 2, i]
        jac = xxi .* yet - xet .* yxi
        
        pg = transpose(shap) * mesh.dgnodes[:, :, i]
        src = forcing.(pg[:, 1], pg[:, 2])
        rs[i] = sum(master.gwgh .* jac .* src)
    end
    
    # Build constraints
    cnstr = zeros(3 * npl1d, nt)
    for i in 1:nt
        for j in 1:3
            f = mesh.t2f[i, j]
            row_offset = (j - 1) * npl1d
            if f > 0
                fpos = f
                ipt = sum(mesh.f[fpos, 1:2])
                ipl = sum(mesh.t[i, :]) - ipt
                isl = findfirst(x -> x == ipl, mesh.t[i, :])
                if mesh.t2f[i, isl] < 0
                    iol = 2
                else
                    iol = 1
                end
                
                xxi = master.sh1d[:, 2, :]' * mesh.dgnodes[perm[:, isl, iol], 1, i]
                yxi = master.sh1d[:, 2, :]' * mesh.dgnodes[perm[:, isl, iol], 2, i]
                dsdxi = sqrt.(xxi.^2 .+ yxi.^2)
                dws = master.gw1d .* dsdxi
                cnstr[(row_offset+1):(row_offset+npl1d), i] = -(sh1d * dws)
            else
                fpos = -f  # Python: -f - 1
                ipt = sum(mesh.f[fpos, 1:2])
                ipr = sum(mesh.t[i, :]) - ipt
                isr = findfirst(x -> x == ipr, mesh.t[i, :])
                if mesh.t2f[i, isr] < 0
                    ior = 2
                else
                    ior = 1
                end
                
                xxi = master.sh1d[:, 2, :]' * mesh.dgnodes[perm[:, isr, ior], 1, i]
                yxi = master.sh1d[:, 2, :]' * mesh.dgnodes[perm[:, isr, ior], 2, i]
                dsdxi = sqrt.(xxi.^2 .+ yxi.^2)
                dws = master.gw1d .* dsdxi
                cnstr[(row_offset+1):(row_offset+npl1d), i] = reverse(sh1d * dws)
            end
        end
    end
    
    # Build big global (face-block) index arrays for the augmented system
    il = zeros(Int, npl1d, npl1d, nf)
    jl = zeros(Int, npl1d, npl1d, nf)
    for i in 1:nf
        row_range = (i-1) * npl1d .+ (1:npl1d)
        il[:, :, i] = repeat(reshape(row_range, :, 1), 1, npl1d)
        jl[:, :, i] = repeat(row_range', npl1d, 1)
    end
    
    # Build index arrays for the constraint part
    ilc = zeros(Int, 3 * npl1d, nt)
    jlc = zeros(Int, 3 * npl1d, nt)
    for i in 1:nt
        for j in 1:3
            f = abs(mesh.t2f[i, j])
            row_offset = (j-1) * npl1d
            f_range = (f-1) * npl1d .+ (1:npl1d)
            ilc[(row_offset+1):(row_offset+npl1d), i] = f_range
        end
        jlc[:, i] .= npl1d * nf + i
    end
    
    M_flat = reshape(M, :)
    il_flat = reshape(il, :)
    jl_flat = reshape(jl, :)
    
    cnstr_flat = reshape(cnstr, :)
    ilc_flat = reshape(ilc, :)
    jlc_flat = reshape(jlc, :)
    
    ilt = [il_flat; ilc_flat; jlc_flat]
    jlt = [jl_flat; jlc_flat; ilc_flat]
    
    Ke = [M_flat; cnstr_flat; cnstr_flat]
    
    rhs_flat = reshape(rhs, :)
    
    F = zeros(npl1d * nf + nt)
    
    F[1:(npl1d*nf)] = rhs_flat
    F[(npl1d*nf+1):end] = rs
    
    K = sparse(ilt, jlt, Ke, npl1d*nf + nt, npl1d*nf + nt)
    
    u = K \ F
    qn = reshape(u[1:(npl1d*nf)], npl1d, nf)
    
    return qn, qn0
end

"""
Given normal equilibrated fluxes, it computes the fluxes q in
the element interior satisfying - Div q = f and evaluates the
lower energy bound.

Parameters:
- master: master structure
- mesh: mesh structure
- forcing: forcing function
- qn: normal gradient to the face

Returns:
- q: reconstructed element fluxes
"""
function reconstruct(master, mesh, forcing, qn)
    nt = size(mesh.t, 1)
    npl = size(mesh.dgnodes, 1)
    ng = size(master.gwgh, 1)
    npl1d = size(master.ploc1d, 1)
    perm = @view master.perm[:, :, 1]
    shap = @view master.shap[:, 1, :]
    sh1d = shape1d(master.porder, master.ploc1d, master.ploc1d[:, 2])

    energy = 0.0
    q = zeros(npl, 2, nt)
   
    # Pre-allocate matrices for reuse
    A = zeros(3 * npl1d + ng, 2 * npl)
    F = zeros(3 * npl1d + ng)
    Z1 = zeros(3 * npl1d + ng, 3 * npl1d + ng)
    F2 = zeros(2*npl + 3*npl1d + ng)
   
    for i in 1:nt
        A .= 0
        F .= 0
       
        for j in 1:3
            f_m = mesh.t2f[i, j]
            row_start = (j - 1) * npl1d + 1
            row_end = j * npl1d
            row_range = row_start:row_end

            if f_m > 0
                f_idx = f_m
                F[row_range] .= qn[:, f_idx]  # Use .= for in-place assignment
            else
                f_idx = -f_m
                F[row_range] .= -reverse(view(qn, :, f_idx))
            end
           
            perm_j = @view perm[:, j]
            dgnodes_j1 = @view mesh.dgnodes[perm_j, 1, i]
            dgnodes_j2 = @view mesh.dgnodes[perm_j, 2, i]
           
            xxi = sh1d[:, 2, :]' * dgnodes_j1
            yxi = sh1d[:, 2, :]' * dgnodes_j2
            dsdxi = sqrt.(xxi.^2 .+ yxi.^2)
           
            # More efficient than creating a full matrix with hcat
            for k in 1:length(perm_j)
                p = perm_j[k]
                nl1 = yxi[k] / dsdxi[k]
                nl2 = -xxi[k] / dsdxi[k]
                A[row_start + k - 1, p] = nl1
                A[row_start + k - 1, p + npl] = nl2
            end
        end

        shap_xi = @view master.shap[:, 2, :]  
        shap_et = @view master.shap[:, 3, :]  
        dgnodes_i1 = @view mesh.dgnodes[:, 1, i]
        dgnodes_i2 = @view mesh.dgnodes[:, 2, i]
       
        xxi = shap_xi' * dgnodes_i1
        xet = shap_et' * dgnodes_i1
        yxi = shap_xi' * dgnodes_i2
        yet = shap_et' * dgnodes_i2
       
        jac = xxi .* yet - xet .* yxi
        wjac = master.gwgh .* jac
        M = shap * Diagonal(wjac) * shap'
       
        # Construct block matrix efficiently
        M2 = [M zeros(size(M));
              zeros(size(M)) M]

        denom = 1.0 ./ jac
        shapx = shap_xi .* (yet .* denom)' - shap_et .* (yxi .* denom)'
        shapy = -shap_xi .* (xet .* denom)' + shap_et .* (xxi .* denom)'

        shap = @view master.shap[:, 1, :]
        pg = shap' * @view mesh.dgnodes[:, :, i]
        src = forcing.(pg[:, 1], pg[:, 2])
       
        row_start = 3 * npl1d + 1
        row_end = row_start + ng - 1
        row_range = row_start:row_end
       
        A[row_range, 1:npl] .= -shapx'
        A[row_range, (npl+1):(2*npl)] .= -shapy'
        F[row_range] .= src
       
        # Construct K and F2 efficiently
        K = [M2 A'; A Z1]
        F2[1:2*npl] .= 0.0
        F2[(2*npl+1):end] .= F
       
        # Solve the system
        sol = pinv(K; rtol=1.e-8) * F2
       
        # Extract solution directly without intermediate allocation
        for k in 1:npl
            q[k, 1, i] = sol[k]
            q[k, 2, i] = sol[k + npl]
        end
       
        # Compute energy contribution
        qx = @view q[:, 1, i]
        qy = @view q[:, 2, i]
        energy += dot(qx, M * qx) + dot(qy, M * qy)
    end
   
    # Final energy
    energy = -0.5 * energy
   
    return q, energy
end