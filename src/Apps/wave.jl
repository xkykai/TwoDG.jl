"""
Calculate interface Roe flux for the wave equation.

# Arguments
- `up`: np plus states, size (np,3)
- `um`: np minus states, size (np,3)
- `nor`: np normal plus vectors, size (np,2)
- `p`: np x,y coordinates, size (np,2)
- `param`: dictionary containing the wave speed c=param[:c]
- `time`: not used

# Returns
- `fn`: np normal fluxes (f plus), size (np,3)
"""
function wavei_roe(up, um, nor, p, param, time)
    c = param[:c]
    ca = abs(c)
    np = size(up, 1)

    # Preallocate arrays
    zer = zeros(np)

    # Vectorized operations with Julia's 1-based indexing
    # Python: fxl = -c*np.column_stack((up[:,2], zer, up[:,0]))
    fxl = -c .* hcat(up[:,3], zer, up[:,1])
    fyl = -c .* hcat(zer, up[:,3], up[:,2])
    fxr = -c .* hcat(um[:,3], zer, um[:,1])
    fyr = -c .* hcat(zer, um[:,3], um[:,2])

    # More efficient implementation of diagonal matrix multiplication
    fav = zeros(np, 3)
    for i in 1:np
        n1 = nor[i,1]
        n2 = nor[i,2]
        for j in 1:3
            fav[i,j] = 0.5 * (n1 * (fxl[i,j] + fxr[i,j]) + n2 * (fyl[i,j] + fyr[i,j]))
        end
    end

    # Calculate flux differences with optimized vector operations
    qb = 0.5 * ca * ((up[:,1] .- um[:,1]) .* nor[:,1] + (up[:,2] .- um[:,2]) .* nor[:,2])
    ub = 0.5 * ca * (up[:,3] .- um[:,3])

    # Pre-allocate result array
    fn = similar(fav)

    # Final calculations
    fn[:,1] .= fav[:,1] .+ qb .* nor[:,1]
    fn[:,2] .= fav[:,2] .+ qb .* nor[:,2]
    fn[:,3] .= fav[:,3] .+ ub

    return fn
end

"""
Calculate the boundary flux for the wave equation

# Arguments
- `up`: np plus states, size (np,3)
- `nor`: np normal plus vectors, size (np,2)
- `ib`: boundary type
            - ib=1: far-field (radiation)
            - ib=2: solid wall(reflection)
            - ib=3: non homogenous far-filed (incoming wave)
- `ui`: infinity state associated with ib
- `p`: np x,y coordinates, size (np,2)
- `param`: dictionary containing
            - the wave speed c=param[:c]
            - the wave vector for incoming waves k=param[:k]
            - the wave function f=param[:f]
- `time`: not used

# Returns
- `fn`: np normal fluxes (f plus), size (np,3)
"""
function waveb(up, nor, ib, ui, p, param, time)
    np = size(up, 1)
    
    if ib == 1  # Far field
        # Julia equivalent of np.matlib.repmat
        um = repeat(ui', np, 1)
    elseif ib == 2  # Reflect
        # Compute normal component of velocity (note Julia 1-based indexing)
        un = up[:,1] .* nor[:,1] .+ up[:,2] .* nor[:,2]
        
        # Reflect velocity components
        um = hcat(up[:,1] .- 2.0 .* un .* nor[:,1], 
                 up[:,2] .- 2.0 .* un .* nor[:,2], 
                 up[:,3])
    elseif ib == 3  # Non-homogenous far-field
        um = zeros(size(up))
        k = param[:k]
        
        # Compute wave function
        um[:,3] = param[:f](param[:c], k, p, time)
        
        # Compute normalized components (note Julia 1-based indexing)
        kmod = sqrt(k[1]^2 + k[2]^2)
        um[:,1] = -k[1] .* um[:,3] ./ kmod
        um[:,2] = -k[2] .* um[:,3] ./ kmod
    else
        error("Invalid boundary type: $ib")
    end
    
    # Extract float parameters for wavei_roe
    roe_param = Dict(:c => param[:c])
    
    # Call the Roe flux function
    fn = wavei_roe(up, um, nor, p, roe_param, time)
    return fn
end

"""
wavev calculate the volume flux for the wave equation.

   u[np,3]:    np left (or plus) states
   p:          not used
   param:      dictionary containing the wave speed c=param["c"]
   time:       not used
   fx[np,3]:   np fluxes in the x direction (f plus)  
   fy[np,3]:   np fluxes in the y direction (f plus)  
"""
function wavev(u, p, param, time)
    zer = zeros(eltype(u), size(u, 1))
    c = param[:c]
    fx = zeros(size(u, 1), 3)
    fy = zeros(size(u, 1), 3)

    fx[:, 1] .= -c .* u[:,3]
    fx[:, 3] .= -c .* u[:,1]

    fy[:, 2] .= -c .* u[:,3]
    fy[:, 3] .= -c .* u[:,2]

    # fx = -c .* hcat(u[:,3], zer, u[:,1])
    # fy = -c .* hcat(zer, u[:,3], u[:,2])
    return fx, fy
end

function mkapp_wave()
    nc = 3
    pg = false
    finvi = wavei_roe
    finvb = waveb
    finvv = wavev

    return App(; nc, pg, finvi, finvb, finvv)
end