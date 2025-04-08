"""
euleri_roe calculate interface roe flux for the euler equations.

   up: left (or plus) states, shape (np,4)
   um: right (or minus) states, shape (np,4)
   nor: normal vectors (pointing outwards the p element), shape (np,2)
   p: x,y coordinates, shape (np,2)
   param: dictionary containing the value of gamma
   time: time
   
Returns:
   fn: normal fluxes (f plus), shape (np,4)  
"""
function euleri_roe(up, um, nor, p, param, time)
    gam = param[:gamma]
    gam1 = gam - 1.0
    
    # Get problem size
    np = size(up, 1)
    
    # Pre-allocate flux array
    fn = zeros(np, 4)
    
    # Extract components (Julia is 1-indexed)                                     
    nx = @view nor[:, 1]
    ny = @view nor[:, 2]
    
    # Extract state variables with views to avoid copying
    rr = @view um[:, 1]
    rur = @view um[:, 2]
    rvr = @view um[:, 3]
    rEr = @view um[:, 4]
    rl = @view up[:, 1]
    rul = @view up[:, 2]
    rvl = @view up[:, 3]
    rEl = @view up[:, 4]
    
    # Compute derived quantities
    rr1 = 1.0 ./ rr
    ur = rur .* rr1
    vr = rvr .* rr1
    Er = rEr .* rr1
    u2r = ur.^2 .+ vr.^2
    pr = gam1 .* (rEr .- 0.5 .* rr .* u2r)
    hr = Er .+ pr .* rr1
    unr = ur .* nx .+ vr .* ny
    
    rl1 = 1.0 ./ rl
    ul = rul .* rl1
    vl = rvl .* rl1
    El = rEl .* rl1
    u2l = ul.^2 .+ vl.^2
    pl = gam1 .* (rEl .- 0.5 .* rl .* u2l)
    hl = El .+ pl .* rl1
    unl = ul .* nx .+ vl .* ny
    
    # Calculate initial flux (first part)
    fn[:, 1] = 0.5 .* (rr .* unr .+ rl .* unl)
    fn[:, 2] = 0.5 .* ((rur .* unr .+ rul .* unl) .+ nx .* (pr .+ pl))
    fn[:, 3] = 0.5 .* ((rvr .* unr .+ rvl .* unl) .+ ny .* (pr .+ pl))
    fn[:, 4] = 0.5 .* (rr .* hr .* unr .+ rl .* hl .* unl)
   
    # Roe average variables
    di = sqrt.(rr .* rl1)    
    d1 = 1.0 ./ (di .+ 1.0)
    ui = (di .* ur .+ ul) .* d1
    vi = (di .* vr .+ vl) .* d1
    hi = (di .* hr .+ hl) .* d1
    ci2 = gam1 .* (hi .- 0.5 .* (ui.^2 .+ vi.^2))
    ci = sqrt.(ci2)
    af = 0.5 .* (ui.^2 .+ vi.^2)
    uni = ui .* nx .+ vi .* ny
    
    # Differences
    dr = rr .- rl
    dru = rur .- rul
    drv = rvr .- rvl
    drE = rEr .- rEl
    
    # Eigenvalues and wave strength coefficients
    rlam1 = abs.(uni .+ ci)
    rlam2 = abs.(uni .- ci)
    rlam3 = abs.(uni)
    s1 = 0.5 .* (rlam1 .+ rlam2)
    s2 = 0.5 .* (rlam1 .- rlam2)
    al1x = gam1 .* (af .* dr .- ui .* dru .- vi .* drv .+ drE)
    al2x = (-uni .* dr) .+ (dru .* nx) .+ (drv .* ny)
    cc1 = ((s1 .- rlam3) .* al1x ./ ci2) .+ (s2 .* al2x ./ ci)
    cc2 = (s2 .* al1x ./ ci) .+ (s1 .- rlam3) .* al2x
     
    # Correct flux with upwinding
    fn[:, 1] = fn[:, 1] .- 0.5 .* (rlam3 .* dr .+ cc1)
    fn[:, 2] = fn[:, 2] .- 0.5 .* (rlam3 .* dru .+ cc1 .* ui .+ cc2 .* nx)
    fn[:, 3] = fn[:, 3] .- 0.5 .* (rlam3 .* drv .+ cc1 .* vi .+ cc2 .* ny)
    fn[:, 4] = fn[:, 4] .- 0.5 .* (rlam3 .* drE .+ cc1 .* hi .+ cc2 .* uni)
    
    return fn
end

"""
eulerb calculate the boundary flux for the euler equations.

   up: np plus states, shape (np,4)
   nor: np normal plus vectors, shape (np,2)
   ib: boundary type
            - ib: 1 far-field (radiation)
            - ib: 2 solid wall(reflection)
            - ib: 3 non homogenous far-filed (incoming wave)
   ui: infinity state associated with ib, length 4
   p: np x,y coordinates, shape (np,2)
   param: dictionary containing the value of gamma
   time: time
   
Returns:
   fn: np normal fluxes (f plus), shape (np,4)  
"""  
function eulerb(up, nor, ib, ui, p, param, time)
    np = size(up, 1)  # Get number of points
    
    if ib == 1                 # Far field
        # Create a matrix with ui repeated np times
        um = repeat(ui', np, 1)
    elseif ib == 2             # Reflect
        # Calculate normal momentum component
        un = up[:, 2] .* nor[:, 1] + up[:, 3] .* nor[:, 2]
        
        # Create reflected state
        um = hcat(
            up[:, 1],                    # Density remains the same
            up[:, 2] .- 2.0 .* un .* nor[:, 1],  # Reflect x-momentum
            up[:, 3] .- 2.0 .* un .* nor[:, 2],  # Reflect y-momentum
            up[:, 4]                     # Energy remains the same
        )
    else  # Handle other cases (including ib==3)
        # Handle as far-field by default
        um = repeat(ui', np, 1)
    end
    
    # Calculate flux using Roe solver
    fn = euleri_roe(up, um, nor, p, param, time)
    
    return fn
end

"""
eulerv calculate the volume flux for the euler equations.

   u[np,4]:      np left (or plus) states
   p:            not used
   param:        dictionary containing the value of gamma
   time:         not used
   fx[np,4]:     np fluxes in the x direction (f plus)  
   fy[np,4]:     np fluxes in the y direction (f plus)  
"""
function eulerv(u::Matrix{T}, p, param, time) where T <: AbstractFloat
    gam = param[:gamma]
    np = size(u, 1)
    
    # Preallocate output arrays
    fx = Matrix{T}(undef, np, 4)
    fy = Matrix{T}(undef, np, 4)
    
    # Compute fluxes
    @inbounds for i in 1:np
        # Extract values
        ρ = u[i,1]  # density
        ρu = u[i,2] # x-momentum
        ρv = u[i,3] # y-momentum
        E = u[i,4]  # total energy
        
        # Velocity components
        uv_i = ρu / ρ
        vv_i = ρv / ρ
        
        # Pressure
        p_i = (gam-1.0) * (E - 0.5 * (ρu * uv_i + ρv * vv_i))
        
        # x-direction fluxes
        fx[i,1] = ρu
        fx[i,2] = ρu * uv_i + p_i
        fx[i,3] = ρv * uv_i
        fx[i,4] = uv_i * (E + p_i)
        
        # y-direction fluxes
        fy[i,1] = ρv
        fy[i,2] = ρu * vv_i
        fy[i,3] = ρv * vv_i + p_i
        fy[i,4] = vv_i * (E + p_i)
    end
    
    return fx, fy
end

"""
eulereval calculates derived quantities for the euler equation variables.

   u[npl,4,nt]:   states
   str:           string used to specify requested quantity
                  - str: "r" density
                  - str: "u" u_x velocity
                  - str: "v" u_y velocity
                  - str: "p" pressure
                  - str: "c" speed of sound
                  - str: "Jp" characteristic variable J+
                  - str: "Jm" characteristic variable J-
                  - str: "M" Mach number
                  - str: "s" entropy
   gam:           value of gamma
Returns:
   sca[npl,nt]:   scalar field requested by str
"""
function eulereval(u::Array{T,3}, str, gam::T) where T<:AbstractFloat
    npl, _, nt = size(u)
    
    if str == "r"
        # Density - just return a view of the first component
        return view(u, :, 1, :)
    end
    
    # Pre-allocate output array for all other cases
    sca = Array{T}(undef, npl, nt)
    
    if str == "u"
        # X-velocity
        @inbounds for j in 1:nt, i in 1:npl
            sca[i, j] = u[i, 2, j] / u[i, 1, j]
        end
    elseif str == "v"
        # Y-velocity
        @inbounds for j in 1:nt, i in 1:npl
            sca[i, j] = u[i, 3, j] / u[i, 1, j]
        end
    else
        # For other quantities, we need the velocity components
        uv = similar(sca)
        vv = similar(sca)
        @inbounds for j in 1:nt, i in 1:npl
            uv[i, j] = u[i, 2, j] / u[i, 1, j]
            vv[i, j] = u[i, 3, j] / u[i, 1, j]
        end
        
        if str == "p"
            # Pressure
            @inbounds for j in 1:nt, i in 1:npl
                sca[i, j] = (gam - 1) * (u[i, 4, j] - 0.5 * (u[i, 2, j] * uv[i, j] + u[i, 3, j] * vv[i, j]))
            end
        else
            # Calculate pressure for other quantities
            p = similar(sca)
            @inbounds for j in 1:nt, i in 1:npl
                p[i, j] = (gam - 1) * (u[i, 4, j] - 0.5 * (u[i, 2, j] * uv[i, j] + u[i, 3, j] * vv[i, j]))
            end
            
            if str == "c"
                # Speed of sound
                @inbounds for j in 1:nt, i in 1:npl
                    sca[i, j] = sqrt(gam * p[i, j] / u[i, 1, j])
                end
            elseif str == "Jp" || str == "Jm"
                # Characteristic variables J+ and J-
                c = similar(sca)
                @inbounds for j in 1:nt, i in 1:npl
                    c[i, j] = sqrt(gam * p[i, j] / u[i, 1, j])
                end
                
                if str == "Jp"
                    @inbounds for j in 1:nt, i in 1:npl
                        sca[i, j] = u[i, 2, j] + 2 * c[i, j] / (gam - 1)
                    end
                else  # str == "Jm"
                    @inbounds for j in 1:nt, i in 1:npl
                        sca[i, j] = u[i, 2, j] - 2 * c[i, j] / (gam - 1)
                    end
                end
            elseif str == "M"
                # Mach number
                @inbounds for j in 1:nt, i in 1:npl
                    u2 = sqrt(uv[i, j]^2 + vv[i, j]^2)
                    sca[i, j] = u2 / sqrt(gam * p[i, j] / u[i, 1, j])
                end
            elseif str == "s"
                # Entropy
                @inbounds for j in 1:nt, i in 1:npl
                    sca[i, j] = p[i, j] / (u[i, 1, j]^gam)
                end
            else
                error("Unknown quantity: $str")
            end
        end
    end
    
    return sca
end

function mkapp_euler()
    nc = 4
    pg = false
    arg = Dict()

    finvi = euleri_roe
    finvb = eulerb
    finvv = eulerv

    return App(; nc, pg, finvi, finvb, finvv, arg)
end