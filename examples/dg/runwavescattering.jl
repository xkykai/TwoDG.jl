using TwoDG
using TwoDG.Masters: shape1d
using Statistics
using CairoMakie

nodetype = 1

# Time parameters for initialization and simulation
time_initialize = 10
time_final = 20
dt = 6e-3
nstep = 20
ncycl_initialize = Int(ceil(time_initialize / (nstep * dt)))  # Number of cycles for initialization
ncycl_equilibrated = Int(ceil((time_final - time_initialize) / (nstep * dt)))  # Cycles after initialization

# Mesh and wave parameters
m = 11
n = 20
c = 1  # Wave speed
k = [3, 0]  # Wave vector
kmod = sqrt(sum(k.^2))  # Magnitude of the wave vector
tparam = [0, 0, 1]  # Trefftz parameter (specific to the Trefftz method)

# Interpolation points for evaluating the solution
interp_points = 0.05:0.1:0.95
n_interp = length(interp_points)

# Polynomial orders for the simulation
porders = [4, 5]
θs_p = []  # Stores angles for each polynomial order
ηs_p = []  # Stores wave amplitudes for each polynomial order

for porder in porders
    @info "Running for porder = $porder"
    mesh = mkmesh_trefftz(m, n, porder, nodetype, tparam)  # Generate the mesh
    master = Master(mesh, 2*porder)  # Create the master element
    shapefunction = shape1d(porder, master.ploc1d, interp_points)  # Shape function for interpolation

    # Find radial points and faces on the mesh boundary
    indp_rad = findall(i -> sqrt(sum(mesh.p[i, :].^2)) ≈ ℯ, 1:size(mesh.p, 1))  # Points on the radial boundary
    indf_rad = findall(i -> mesh.f[i, 1] in indp_rad && mesh.f[i, 2] in indp_rad, 1:size(mesh.f, 1))  # Faces on the radial boundary

    ng1d = length(master.gw1d)  # Number of Gauss points in 1D
    θs = zeros(length(indf_rad) * n_interp)  # Array to store angles
    ηs = similar(θs)  # Array to store wave amplitudes
    permls = zeros(Int, length(indf_rad), size(master.ploc1d, 1))  # Permutation indices for each face
    els = zeros(Int, length(indf_rad))  # Element indices for each face

    iθ = 1
    for (i, face_num) in enumerate(indf_rad)
        ipt = mesh.f[face_num, 1] + mesh.f[face_num, 2]  # Sum of point indices for the face
        el = mesh.f[face_num, 3]  # Element index associated with the face

        ipl = sum(mesh.t[el, :]) - ipt  # Opposite point index in the element
        isl = findfirst(x -> x == ipl, mesh.t[el, :])  # Local index of the opposite point
        iol = mesh.t2f[el, isl] < 0 ? 2 : 1  # Orientation of the face (adjusted for Julia's 1-based indexing)
        perml = master.perm[:, isl, iol]  # Permutation indices for the face
        dgs = mesh.dgnodes[perml, :, el]  # DG nodes for the face

        dgs_interp = shapefunction[:, 1, :]' * dgs  # Interpolated DG nodes

        θ = atan.(dgs_interp[:, 2], dgs_interp[:, 1])  # Compute angles for the face

        θs[iθ:iθ + n_interp - 1] .= θ  # Store angles
        permls[i, :] .= perml  # Store permutation indices
        els[i] = el  # Store element index
        iθ += n_interp
    end

    θs[θs .< 0] .+= 2π  # Adjust angles to be in the range [0, 2π]

    # Boundary condition setup
    bcm = [2, 3]  # Boundary condition markers
    bcs = zeros(3, 3)  # Boundary condition values
    app = mkapp_wave()  # Create wave application
    app = App(app; bcm, bcs)  # Apply boundary conditions

    # Define the wave function
    ub(c, k, p, time) = sin.(p[:,1] .* k[1] .+ p[:,2] .* k[2] .- c * sqrt(k[1]^2 + k[2]^2) * time)

    # Set application parameters
    app.pg = true
    app.arg[:c] = c
    app.arg[:k] = k
    app.arg[:f] = ub

    # Initialize solution arrays
    u = initu(mesh, app, [0, 0, 0])  # Initial solution
    ue = zeros(size(u))  # Exact solution

    @info "Initializing the solution..."
    tm = 0  # Time variable
    for i in 1:ncycl_initialize
        # Update `ue` based on the current time `tm`
        ue[:, 3, :] .= sin.(mesh.dgnodes[:, 1, :] .* k[1] .+ mesh.dgnodes[:, 2, :] .* k[2] .- c * kmod * tm)
        ue[:, 1, :] .= -k[1] * ue[:, 3, :] / kmod
        ue[:, 2, :] .= -k[2] * ue[:, 3, :] / kmod

        # Set initial condition for `u` if it's the first iteration
        if i == 1
            u .= ue
        end

        rk4!(rinvexpl, master, mesh, app, u, tm, dt, nstep)  # Runge-Kutta time-stepping

        # Increment time
        tm += nstep * dt
    end

    @info "Computing wave amplitude..."
    for i in 1:ncycl_equilibrated
        # Update `ue` based on the current time `tm`
        ue[:, 3, :] .= sin.(mesh.dgnodes[:, 1, :] .* k[1] .+ mesh.dgnodes[:, 2, :] .* k[2] .- c * kmod * tm)
        ue[:, 1, :] .= -k[1] * ue[:, 3, :] / kmod
        ue[:, 2, :] .= -k[2] * ue[:, 3, :] / kmod

        rk4!(rinvexpl, master, mesh, app, u, tm, dt, nstep)  # Runge-Kutta time-stepping

        iθ = 1
        for (i, face_num) in enumerate(indf_rad)
            el = els[i]
            perml = permls[i, :]
            us = u[perml, 3, el]  # Extract solution for the face
            ηs_interp = abs.(shapefunction[:, 1, :]' * us)  # Interpolated wave amplitude

            η = @view ηs[iθ:iθ + n_interp - 1]  # View into the wave amplitude array
            η .= (ηs_interp .> η) .* ηs_interp .+ (ηs_interp .<= η) .* η  # Update wave amplitude

            iθ += n_interp
        end

        # Increment time
        tm += nstep * dt
    end

    # Store results for plotting
    push!(θs_p, θs)
    push!(ηs_p, ηs)
end

# Plot the results
fig = Figure()
ax = Axis(fig[1, 1], xlabel="θ / π", ylabel="|η|", title="Scattered wave amplitude")
for (i, porder) in enumerate(porders)
    θ_perm = sortperm(θs_p[i])  # Sort angles for plotting
    lines!(ax, θs_p[i][θ_perm] ./ π, abs.(ηs_p[i][θ_perm]), label="p = $(porder)")
end
axislegend(ax, position=:ct)
display(fig)
# save("./output/scattered_wave_amplitude.png", fig, px_per_unit=8)