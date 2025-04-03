"""
    rk4(residexpl, master, mesh, app, u, time, dt, nstep)

Optimized RK4 time integrator using a 4 stage Runge-Kutta scheme.

# Arguments
- `residexpl`: Function for residual evaluation `r = residexpl(master, mesh, app, u, time)`
- `master`: Master structure
- `mesh`: Mesh structure
- `app`: Application structure
- `u`: Vector of unknowns with dimensions (npl, nc, nt)
- `time`: Current time
- `dt`: Time step
- `nstep`: Number of steps to be performed

# Returns
- Updated `u` after `nstep` steps
- Final time value
"""
function rk4(residexpl::Function, master, mesh, app, u, time::Real, dt::Real, nstep::Integer)
    # Create a copy to avoid modifying the input
    u_current = copy(u)
    t_current = time
    
    # Pre-allocate temporary arrays for efficiency
    temp = similar(u)
    
    for i in 1:nstep
        # Calculate k1
        k1 = dt .* residexpl(master, mesh, app, u_current, t_current)
        
        # Calculate k2
        @. temp = u_current + 0.5 * k1
        k2 = dt .* residexpl(master, mesh, app, temp, t_current + 0.5*dt)
        
        # Calculate k3
        @. temp = u_current + 0.5 * k2
        k3 = dt .* residexpl(master, mesh, app, temp, t_current + 0.5*dt)
        
        # Calculate k4
        @. temp = u_current + k3
        k4 = dt .* residexpl(master, mesh, app, temp, t_current + dt)
        
        # Update u_current
        @. u_current += (k1/6 + k2/3 + k3/3 + k4/6)
        
        t_current += dt
    end
    
    return u_current
end

"""
    rk4!(residexpl, master, mesh, app, u, time, dt, nstep)

In-place RK4 time integrator using a 4 stage Runge-Kutta scheme.

# Arguments
- `residexpl`: Function for residual evaluation `r = residexpl(master, mesh, app, u, time)`
- `master`: Master structure
- `mesh`: Mesh structure
- `app`: Application structure
- `u`: Vector of unknowns, modified in-place
- `time`: Current time
- `dt`: Time step
- `nstep`: Number of steps to be performed

# Returns
- `u`: Updated after `nstep` steps (modified in-place)
- Final time value
"""
function rk4!(residexpl::Function, master, mesh, app, u, time::Real, dt::Real, nstep::Integer)
    t_current = time
    
    # Pre-allocate temporary arrays for efficiency
    temp = similar(u)
    
    for i in 1:nstep
        # Calculate k1
        k1 = dt .* residexpl(master, mesh, app, u, t_current)
        
        # Calculate k2
        @. temp = u + 0.5 * k1
        k2 = dt .* residexpl(master, mesh, app, temp, t_current + 0.5*dt)
        
        # Calculate k3
        @. temp = u + 0.5 * k2
        k3 = dt .* residexpl(master, mesh, app, temp, t_current + 0.5*dt)
        
        # Calculate k4
        @. temp = u + k3
        k4 = dt .* residexpl(master, mesh, app, temp, t_current + dt)
        
        # Update u in-place
        @. u += (k1/6 + k2/3 + k3/3 + k4/6)
        
        t_current += dt
    end
    
    return u
end