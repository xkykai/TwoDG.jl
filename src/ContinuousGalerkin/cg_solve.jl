using SparseArrays
using LinearAlgebra
using Statistics

function cg_solve(mesh, master, source, param)
    """
    cg_solve solves the convection-diffusion equation using the cg method.
    [uh,energy]=cg_solve(mesh,master,source,param)
 
        master:       master structure
        mesh:         mesh structure
        source:       source term
        param:        kappa:   = diffusivity coefficient
                        c      = convective velocity
                        s      = source coefficient
        u:            approximate scalar variable
        uh:           approximate scalar variable with local numbering
    """
   
    npl = size(mesh.plocal, 1)
    nt = size(mesh.tcg, 1)
    nn = size(mesh.pcg, 1)
    
    ae = Array{Float64}(undef, npl, npl, nt)
    fe = Array{Float64}(undef, npl, nt)
    
    for i in 1:nt
        A, F = elemmat_cg(mesh.pcg[mesh.tcg[i,:],:], master, source, param)
        ae[:,:,i] .= A
        fe[:,i] .= F
    end

    # Dirichlet boundary conditions
    bou = zeros(Int, size(mesh.pcg, 1))
    ii = findall(x -> x < 0, mesh.f[:,4])  # Python's mesh.f[:,3] → Julia's mesh.f[:,4]
    
    for i in ii
        el = mesh.f[i,3]  # Python's mesh.f[i,2] → Julia's mesh.f[i,3]
        ipl = sum(mesh.t[el, :]) - sum(mesh.f[i, 1:2])
        isl = findall(x -> x == ipl, mesh.t[el, :])
        
        # Vector-based assignment matching Python's behavior
        bou[mesh.tcg[el, master.perm[:,isl,1]]] .= 1
    end
    
    for i in 1:nt
        for j in 1:npl
            if bou[mesh.tcg[i,j]] == 1
                ae[j, :, i] .= 0.0
                ae[j, j, i] = 1.0
                fe[j, i] = 0.0
            end
        end
    end
    
    K = spzeros(nn, nn)
    F = zeros(nn, 1)  # Make F a column vector to match Python
    
    for i in 1:nt
        elem = mesh.tcg[i,:]
        for j in 1:npl, k in 1:npl
            K[elem[j], elem[k]] += ae[j, k, i]
        end
        F[elem, 1] .+= fe[:, i]  # Index into first column
    end
    
    u = K \ F  # Julia's backslash operator
    energy = 0.5 .* u' * K * u .- u' * F
    
    # Output uh (DG format) to make it compatible with scaplot
    uh = Array{Float64}(undef, size(mesh.dgnodes, 1), size(mesh.dgnodes, 3))
    
    for i in 1:size(mesh.tcg, 1)
        uh[:,i] .= u[mesh.tcg[i,:]]
    end
    
    # Extract scalar value from energy if it's an array
    energy_scalar = isa(energy, Array) ? energy[1] : energy
    
    return uh, energy_scalar, u
end