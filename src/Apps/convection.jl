"""
convectioni calculate interface upwind flux for the linear convection equation.
   up:     plus states
   um:     minus states
   nor:    normal plus vectors
   p:      x,y coordinates
   param:  dictionary containing either
           - a constant velocity field [u,v] = param[:vf]
           - a function that returns a velocity field as a function of p vvec = param[:vf](p)
   time:   not used
   returns: normal fluxes (f plus)    
"""
function convectioni(up, um, nor, p, param, time)
    # Get velocity field and compute normal velocity component
    if param[:vf] isa Function
        vfield = param[:vf](p)
        vn = vec(sum(nor .* vfield, dims=2))  # Dot product along each row
    else
        vfield = param[:vf]
        vn = nor[:,1] .* vfield[1] .+ nor[:,2] .* vfield[2]
    end

    # Compute absolute value of normal velocity
    avn = abs.(vn)
    
    # Reshape for proper broadcasting if needed
    if ndims(up) > 1
        vn = reshape(vn, :, 1)
        avn = reshape(avn, :, 1)
    end
    
    # Compute flux using upwind scheme
    fn = 0.5 .* vn .* (up .+ um) .+ 0.5 .* avn .* (up .- um)
    
    return fn
end

"""
convectionb calculate the boundary flux for the linear convection equation.

   up:     plus states
   nor:    normal plus vectors (pointing outwards the p element)
   ib:     boundary type
           - ib: 1 far-field (radiation)
   ui:     infinity state associated with ib
   p:      x,y coordinates
   param:  dictionary containing either
           - a constant velocity field [u,v] = param[:vf]
           - a function that returns a velocity field as a function of p vvec = param[:vf](p)
   time:   not used
   returns: normal fluxes (f plus)  
""" 
function convectionb(up, nor, ib, ui, p, param, time)
    # Replicate boundary state to match shape of interior state
    if ndims(up) == 1
        um = fill(ui[1], size(up))
    else
        um = repeat(ui, outer=(size(up,1), 1))
    end
    
    # Call convectioni to compute flux
    fn = convectioni(up, um, nor, p, param, time)
    
    return fn
end

"""
convectionv calculate the volume flux for the linear convection equation.

   u:      left (or plus) states
   p:      x,y coordinates
   param:  dictionary containing either
           - a constant velocity field [u,v] = param[:vf]
           - a function that returns a velocity field as a function of p vvec = param[:vf](p)
   time:   not used
   returns: tuple of (fx, fy) where
            fx: fluxes in the x direction (f plus)  
            fy: fluxes in the y direction (f plus)  
"""
function convectionv(u, p, param, time)
    if param[:vf] isa Function
        vfield = param[:vf](p)
        
        # Compute fluxes using broadcasting
        fx = vfield[:,1] .* u
        fy = vfield[:,2] .* u
    else
        vfield = param[:vf]
        fx = vfield[1] .* u
        fy = vfield[2] .* u
    end
    
    return fx, fy
end

function mkapp_convection()
    pg = true
    finvi = convectioni
    finvb = convectionb
    finvv = convectionv

    return App(; nc=1, pg, finvi, finvb, finvv)
end
