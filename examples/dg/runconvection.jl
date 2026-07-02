using TwoDG
using Statistics
using CairoMakie

# Total simulation time and time step size
time_total = 2π
dt = 2π / 1000
nstep = 25  # Number of steps per RK4 iteration
parity = 0  # Parity of the mesh (used in mesh generation)
nodetype = 1  # Type of nodes in the mesh
niter = Int(1000 / 25)  # Total number of iterations

# Grid sizes and polynomial orders for convergence study
ngrids = [9, 13, 17]  # Number of grid points in each dimension
ps = [2, 3, 4]  # Polynomial orders for DG method

# Arrays to store L2 errors for regular and distorted meshes
l2_errors = [zeros(length(ngrids)) for _ in ps]
l2_errors_distorted = [zeros(length(ngrids)) for _ in ps]

# Loop over grid sizes and polynomial orders
for (i, ngrid) in enumerate(ngrids), (j, p) in enumerate(ps)
    m = ngrid
    n = ngrid
    porder = p  # Polynomial order for DG method

    # Generate a square mesh
    mesh = mkmesh_square(m, n, porder, parity, nodetype)
    master = Master(mesh, 2*porder)  # Create master element for DG method

    # Create a distorted version of the mesh
    mesh_distorted = deepcopy(mesh)
    mkmesh_distort!(mesh_distorted)  # Apply distortion to the mesh

    # Boundary conditions
    bcm = ones(Int64, 4)  # Boundary condition markers
    bcs = zeros(1, 1)  # Boundary condition values

    # Velocity field function
    vf(p) = hcat(-p[:, 2] .+ 0.5, p[:, 1] .- 0.5)

    # Create application object for convection problem
    app = mkapp_convection()
    app = App(app; bcm, bcs)
    app.arg[:vf] = vf  # Assign velocity field to the application

    # Initial condition function
    init(x, y) = exp(-120 * ((x - 0.6)^2 + (y - 0.5)^2))

    # Initialize solution on the regular mesh
    u = initu(mesh, app, [init])
    @info "Computing MSE for p = $(p) and size = $(ngrids[i])"

    tm = 0.  # Initialize time
    for i in 1:niter
        # Update the solution using the RK4 time-stepping method
        rk4!(rinvexpl, master, mesh, app, u, tm, dt, nstep)
        tm += nstep * dt  # Increment time
    end
    # Compute L2 error for the regular mesh
    l2_errors[j][i] = l2_error(mesh, u, init)

    @info "Computing MSE for distorted mesh p = $(p) and size = $(ngrids[i])"

    # Initialize solution on the distorted mesh
    u_distorted = initu(mesh_distorted, app, [init])
    tm = 0.  # Reset time
    for i in 1:niter
        # Update the solution using the RK4 time-stepping method
        rk4!(rinvexpl, master, mesh_distorted, app, u_distorted, tm, dt, nstep)
        tm += nstep * dt  # Increment time
    end
    # Compute L2 error for the distorted mesh
    l2_errors_distorted[j][i] = l2_error(mesh_distorted, u_distorted, init)
end

# Plotting error convergence for regular mesh
fig = Figure()
ax = Axis(fig[1, 1], xlabel="1/h", ylabel="L2 norm", title="Error convergence of DG explicit linear convection", yscale=log10, xscale=log2)
for i in eachindex(ngrids)
    scatterlines!(ax, ngrids .- 1, l2_errors[i], label="p=$(ps[i])")  # Plot errors for each polynomial order
end
axislegend(ax, position=:lb)
display(fig)
# save("./output/dg_convection_convergence.png", fig, px_per_unit=8)  # Save the plot

# Plotting error convergence for distorted mesh
fig = Figure()
ax = Axis(fig[1, 1], xlabel="1/h", ylabel="L2 norm", title="Error convergence of DG explicit linear convection (distorted mesh)", yscale=log10, xscale=log2)
for i in eachindex(ngrids)
    scatterlines!(ax, ngrids .- 1, l2_errors_distorted[i], label="p=$(ps[i])")  # Plot errors for each polynomial order
end
axislegend(ax, position=:lb)
display(fig)
# save("./output/dg_convection_convergence_distorted.png", fig, px_per_unit=8)  # Save the plot