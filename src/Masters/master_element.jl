using LinearAlgebra
using TwoDG.Meshes: Mesh

struct Master{PO, PL, C, PE, PLO, GP, GPT, GW, GWG, SH, SHA}
    porder::PO
    plocal::PL
    corner::C
    perm::PE
    ploc1d::PLO
    gp1d::GP
    gpts::GPT
    gw1d::GW
    gwgh::GWG
    sh1d::SH
    shap::SHA
end

function Master(mesh::Mesh)
    # Initialize Master with basic fields
    master = Master(mesh.porder, mesh.plocal, zeros(Int, 3), zeros(Int, mesh.porder+1, 3, 2), nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    
    # Find corners
    for i in 1:3
        master.corner[i] = findfirst(x -> x > 1-1e-6, master.plocal[:, i])
    end
    
    # Process permutations
    aux = [1, 2, 3, 1, 2]
    ploc1d = nothing
    
    for i in 1:3
        ii = findall(x -> x < 1e-6, master.plocal[:, i])
        jj = sortperm(master.plocal[ii, aux[i+2]])
        master.perm[:, i, 1] = ii[jj]
        
        if i == 3
            ploc1d = master.plocal[ii[jj], 1:2]
        end
    end
    
    # Flip permutation
    master.perm[:,:,2] = reverse(master.perm[:,:,1], dims=1)
    
    # Compute Gaussian quadrature points and weights
    pgauss = max(4*mesh.porder, 1)
    gp1d, gw1d = gaussquad1d(pgauss)
    gpts, gwgh = gaussquad2d(pgauss)

    sh1d = shape1d(master.porder, ploc1d, gp1d)
    shap = shape2d(master.porder, master.plocal, gpts)
    
    # Create and return the final Master object
    return Master(master.porder, master.plocal, master.corner, master.perm, ploc1d, gp1d, gpts, gw1d, gwgh, sh1d, shap)
end

"""
uniformlocalpnts 2-d mesh generator for the master element.
[plocal,tlocal]=uniformlocalpnts(porder)

   plocal:    node positions (npl,3)
   tlocal:    triangle indices (nt,3)
   porder:    order of the complete polynomial 
              npl = (porder+1)*(porder+2)/2
 """
function uniformlocalpnts(porder)
    plocal = zeros()
    n = porder + 1
    npl = (porder + 1) * (porder + 2) ÷ 2

    plocal = zeros(npl, 3)
    xs = ys = range(0, 1, length=n)

    i_start = 1
    for i in 1:n
        i_end = i_start + n - i
        plocal[i_start:i_end, 2] .= xs[1:n+1-i]
        plocal[i_start:i_end, 3] .= ys[i]
        plocal[i_start:i_end, 1] .= xs[n+1-i:-1:1]
        i_start = i_end + 1
    end

    tlocal = zeros(Int, porder^2, 3)
    i_start_t = 1
    vertex_start = 1
    for i in 1:porder
        i_end_t = i_start_t + porder - i
        tlocal[i_start_t:i_end_t, 1] .= vertex_start:vertex_start + porder - i
        tlocal[i_start_t:i_end_t, 2] .= vertex_start + 1:vertex_start + porder - i + 1
        tlocal[i_start_t:i_end_t, 3] .= vertex_start + porder - i + 2:vertex_start + 2porder - 2i + 2
        
        i_start_t = i_end_t + 1

        if i_start_t < porder^2
            i_end_t = i_start_t + porder - i - 1
            vertex_start += 1
            tlocal[i_start_t:i_end_t, 1] .= vertex_start:vertex_start + porder - i - 1
            tlocal[i_start_t:i_end_t, 2] .= vertex_start + porder - i + 2:vertex_start + 2porder - 2i + 1
            tlocal[i_start_t:i_end_t, 3] .= vertex_start + porder - i + 1:vertex_start + 2porder - 2i
            i_start_t = i_end_t + 1
        end
        
        vertex_start += porder - i + 1
    end

    return plocal, tlocal
end

"""     
shape1d calculates the nodal shapefunctions and its derivatives for 
         the master 1d element [0,1]
nfs=shape1d(porder,plocal,pts)

   porder:    polynomial order
   plocal:    node positions (np) (np=porder+1)
   pts:       coordinates of the points where the shape fucntions
               and derivatives are to be evaluated (npoints)
   nsf:       shape function adn derivatives (np,2,npoints)
              nsf[:,0,:] shape functions 
              nsf[:,1,:] shape fucntions derivatives w.r.t. x 
"""
function shape1d(porder, plocal, pts)
    # Number of nodes
    np = porder + 1
    # Number of evaluation points
    npoints = length(pts)
    
    # Extract x-coordinates from plocal and use them for Koornwinder
    x_coords = plocal[:, 1]
    
    # Create Vandermonde matrix at node locations
    V, Vξ = koornwinder1d(x_coords, porder)
    
    # Calculate coefficient matrix A
    A = (V \ I)
    
    # Initialize output array for shape functions and derivatives
    nfs = zeros(np, 2, npoints)
    
    # For each evaluation point
    Λ, dΛ = koornwinder1d(pts, porder)
    
    nfs[:, 1, :] = (Λ * A)'
    nfs[:, 2, :] = (dΛ * A)'
    
    return nfs
end

"""     
shape2d calculates the nodal shapefunctions and its derivatives for 
        the master triangle [0,0]-[1,0]-[0,1]
nfs=shape2d(porder,plocal,pts)

porder:    polynomial order
plocal:    node positions (np,2) (np=(porder+1)*(porder+2)/2)
pts:       coordinates of the points where the shape fucntions
             and derivatives are to be evaluated (npoints,2)
nfs:       shape function adn derivatives (np,3,npoints)
             nsf[:,0,:] shape functions 
             nsf[:,1,:] shape fucntions derivatives w.r.t. x
             nsf[:,2,:] shape fucntions derivatives w.r.t. y
"""
function shape2d(porder, plocal, pts)
    # Calculate number of nodes for this polynomial order
    np = (porder + 1) * (porder + 2) ÷ 2
    
    # Number of evaluation points
    npoints = size(pts, 1)
    
    # Calculate coefficient matrix A
    W, _, _ = koornwinder2d(plocal[:, 2:3], porder)
    
    # display(W)
    A = W \ I

    # Initialize output array for shape functions and derivatives
    nfs = zeros(np, 3, npoints)
    
    # Evaluate at all points
    Λ, Λξ, Λη = koornwinder2d(pts, porder)

    ϕ = (Λ * A)'
    ϕξ = (Λξ * A)'
    ϕη = (Λη * A)'

    nfs[:, 1, :] .= ϕ
    nfs[:, 2, :] .= ϕξ
    nfs[:, 3, :] .= ϕη
    
    return nfs
end

function get_local_face_nodes(mesh, master, face_number, flip_face_direction=false)
    it = mesh.f[face_number, 3]
    local_face_number = findfirst(x -> x == face_number, mesh.t2f[it, :])
    if flip_face_direction
        node_numbers = master.perm[:, local_face_number, 2]
    else
        node_numbers = master.perm[:, local_face_number, 1]
    end

    return node_numbers
end