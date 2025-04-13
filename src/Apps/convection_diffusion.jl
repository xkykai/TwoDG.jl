"""
    cdinvv(u, p, param, time)

Calculate the volume flux for the linear convection-diffusion equation.

# Arguments
- `u`: Array of left (or plus) states
- `p`: Matrix of x,y coordinates
- `param`: Dictionary containing velocity field information
- `time`: Not used

# Returns
- `fx`: Fluxes in the x direction
- `fy`: Fluxes in the y direction
"""
function cdinvv(u, p, param, time)
    vfield = param[:vf](p)
    fx = vfield[:,1] .* u
    fy = vfield[:,2] .* u

    return fx, fy
end

"""
    cdinvi(up, um, nor, p, param, time)

Calculate interface upwind flux for the linear convection-diffusion equation.

# Arguments
- `up`: Plus states
- `um`: Minus states
- `nor`: Normal plus vectors
- `p`: Matrix of x,y coordinates
- `param`: Dictionary containing velocity field information
- `time`: Not used

# Returns
- `fn`: Normal fluxes
"""
function cdinvi(up, um, nor, p, param, time)
    vfield = param[:vf](p)
    vn = sum(vfield .* nor, dims=2)
    avn = abs.(vn)
    fn = 0.5 .* vn .* (up .+ um) .+ 0.5 .* avn .* (up .- um)

    return fn
end

"""
    cdinvb(up, nor, ib, ui, p, param, time)

Calculate the boundary flux for the linear convection-diffusion equation.

# Arguments
- `up`: Plus states
- `nor`: Normal plus vectors (pointing outwards the p element)
- `ib`: Boundary type (0: Dirichlet, 1: Neumann)
- `ui`: Infinity state associated with ib
- `p`: Matrix of x,y coordinates
- `param`: Dictionary containing velocity field information
- `time`: Not used

# Returns
- `fn`: Normal fluxes
"""
function cdinvb(up, nor, ib, ui, p, param, time)
    if ib == 0      # Dirichlet
        um = zero(up)
    elseif ib == 1  # Neumann
        um = copy(up)
    end

    fn = cdinvi(up, um, nor, p, param, time)

    return fn
end

"""
    cdvisi(up, um, qp, qm, nor, p, param, time)

Calculate the viscous interface upwind flux for the linear convection-diffusion equation.

# Arguments
- `up`: Plus states
- `um`: Minus states
- `qp`: Plus q states
- `qm`: Minus q states
- `nor`: Normal plus vectors
- `p`: Matrix of x,y coordinates
- `param`: Dictionary containing parameters
- `time`: Not used

# Returns
- `fn`: Normal fluxes
"""
function cdvisi(up, um, qp, qm, nor, p, param, time)
    kappa = param[:kappa]
    c11int = param[:c11int]
    fn = -kappa .* (qm[:,1] .* nor[:,1] .+ qm[:,2] .* nor[:,2]) .+ c11int .* (up .- um)

    return fn
end

"""
    cdvisb(up, qp, nor, ib, ui, p, param, time)

Calculate the viscous boundary flux for the convection-diffusion equation.

# Arguments
- `up`: Plus states
- `qp`: Plus q states
- `nor`: Normal plus vectors (pointing outwards the p element)
- `ib`: Boundary type (0: Dirichlet, 1: Neumann)
- `ui`: Infinity state associated with ib
- `p`: Matrix of x,y coordinates
- `param`: Dictionary containing parameters
- `time`: Not used

# Returns
- `fn`: Normal fluxes
"""
function cdvisb(up, qp, nor, ib, ui, p, param, time)
    kappa = param[:kappa]
    c11 = param[:c11]
    
    if ib == 0      # Dirichlet
        fn = -kappa .* (qp[:,1] .* nor[:,1] .+ qp[:,2] .* nor[:,2]) .+ c11 .* (up .- ui)
    elseif ib == 1  # Neumann
        fn = zeros(eltype(up), size(up))
    end

    return fn
end

"""
    cdvisv(u, q, p, param, time)

Calculate the viscous volume flux for the linear convection-diffusion equation.

# Arguments
- `u`: States
- `q`: Gradient states
- `p`: Matrix of x,y coordinates
- `param`: Dictionary containing parameters
- `time`: Not used

# Returns
- `fx`: Fluxes in the x direction
- `fy`: Fluxes in the y direction
"""
function cdvisv(u, q, p, param, time)
    kappa = param[:kappa]
    
    fx = -kappa .* q[:,1]
    fy = -kappa .* q[:,2]

    return fx, fy
end

"""
    cdvisub(up, nor, ib, ui, p, param, time)

Calculate the viscous boundary flux for the convection-diffusion equation.

# Arguments
- `up`: Plus states
- `nor`: Normal plus vectors (pointing outwards the p element)
- `ib`: Boundary type (0: Dirichlet, 1: Neumann)
- `ui`: Infinity state associated with ib
- `p`: Matrix of x,y coordinates
- `param`: Dictionary containing parameters
- `time`: Not used

# Returns
- `ub`: Values of u at the boundary interface
"""
function cdvisub(up, nor, ib, ui, p, param, time)
    if ib == 0      # Dirichlet
        ub = zero(up)
    elseif ib == 1  # Neumann
        ub = copy(up)
    end

    return ub
end

"""
mkapp create application structure template for the linear convection-diffusion equation.
   app:   application structure
"""
function mkapp_convection_diffusion()
    nc = 1
    pg = true
    
    finvi = cdinvi
    finvb = cdinvb
    finvv = cdinvv
    fvisi = cdvisi
    fvisb = cdvisb
    fvisv = cdvisv
    fvisub = cdvisub


    return App(; nc, pg, finvi, finvb, finvv, fvisi, fvisb, fvisv, fvisub)
end
