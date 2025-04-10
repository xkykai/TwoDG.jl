using TwoDG
using CairoMakie

# Mesh and polynomial order parameters
m = 16  # Number of elements in the x-direction
n = 8   # Number of elements in the y-direction
porder = 5  # Polynomial order for the finite element method

# Mesh configuration parameters
parity = 1
nodetype = 1
gam = 1.4  # Specific heat ratio (gamma) for the gas
Minf = 0.3  # Freestream Mach number

# Figure size for visualization
figure_size = (1000, 400)

# Initial conditions for the flow variables
ui = [1.0, 1.0, 0, 0.5 + 1 / (gam * (gam - 1) * Minf^2)]  # [rho, rho*u, rho*v, rho*E]

# Duct geometry parameters
db = 0  # Bottom wall displacement
dt = 0  # Top wall displacement
H = 1   # Height of the duct

# Create a square mesh and modify it to represent a duct
mesh = mkmesh_square(m, n, porder, parity, nodetype)
master = Master(mesh, 2*porder)  # Master element for higher-order methods
mesh = mkmesh_duct(mesh, db, dt, H)  # Transform the square mesh into a duct
mesh.fcurved .= false  # Disable curved faces
mesh.tcurved .= false  # Disable curved elements

# Boundary conditions
bcm = [2, 1, 2, 1]  # Boundary condition types for each side of the domain
bcs = vcat(ui', 0.0 .* ui')  # Boundary condition values

# Application setup for Euler equations
app = mkapp_euler()
app = App(app; bcm, bcs)
app.arg[:gamma] = gam  # Set the specific heat ratio in the application

# Background flow properties
background_rho = 1  # Background density
background_ru = 1 * ui[2] / ui[1]  # Background momentum in x-direction
background_rv = 1 * ui[3] / ui[1]  # Background momentum in y-direction
pinf = 1 / (gam * Minf^2)  # Background pressure
background_rE = 0.5 * (background_ru^2 + background_rv^2) / background_rho + pinf / (gam - 1)  # Background energy
background_riemann = canonical_to_riemann(background_rho, background_ru, background_rv, background_rE, gam)

# Derived background properties
background_pressure = (gam - 1) * (background_rE - 0.5 * (background_ru^2 + background_rv^2) / background_rho)
background_c = sqrt(gam * background_pressure / background_rho)  # Background speed of sound

# Plane wave perturbation function
plane_wave(x, y) = x >= 1.0 && x <= 2.0 ? 0.1 * sin(2π * (x-1)) : 0  # Perturbation in a specific region

# Initial conditions for Riemann invariants
riemann_initv(x, y) = background_riemann[1]  # Initial velocity
riemann_inits(x, y) = background_riemann[2]  # Initial entropy
riemann_initJ⁺_wave(x, y) = background_riemann[3] + plane_wave(x, y)  # J⁺ with wave perturbation
riemann_initJ⁺(x, y) = background_riemann[3]  # J⁺ without wave perturbation
riemann_initJ⁻_wave(x, y) = background_riemann[4] + plane_wave(x, y)  # J⁻ with wave perturbation
riemann_initJ⁻(x, y) = background_riemann[4]  # J⁻ without wave perturbation

# Initialize the solution for J⁺
u_riemann_J⁺ = initu(mesh, app, [riemann_initv, riemann_inits, riemann_initJ⁺_wave, riemann_initJ⁻])
u_canonical_J⁺ = similar(u_riemann_J⁺)  # Allocate memory for the canonical variables

# Convert Riemann invariants to canonical variables
for i in axes(u_riemann_J⁺, 1), k in axes(u_riemann_J⁺, 3)
    u_canonical_J⁺[i, :, k] .= riemann_to_canonical(u_riemann_J⁺[i, 1, k], u_riemann_J⁺[i, 2, k], u_riemann_J⁺[i, 3, k], u_riemann_J⁺[i, 4, k], gam)
end

# Time-stepping parameters
Δt = 2e-4  # Time step size
nstep = 20  # Number of steps per cycle
tm = 0.  # Initial time
time_total = 0.24  # Total simulation time
ncycl = Int(ceil(time_total / Δt / nstep))  # Number of cycles

# Plot initial condition for J⁺
fig = scaplot(mesh, eulereval(u_canonical_J⁺, "Jp", gam), show_mesh=true, figure_size=figure_size, title="J⁺, t = $(round(tm, sigdigits=3))")
# save("./output/Jp_initial.png", fig, px_per_unit=8)

# Time-stepping loop for J⁺
for i in 1:ncycl
    @info "$tm"  # Log the current time
    rk4!(rinvexpl, master, mesh, app, u_canonical_J⁺, tm, Δt, nstep)  # Advance the solution using RK4
    tm += nstep * Δt  # Update time
end

# Plot final condition for J⁺
fig = scaplot(mesh, eulereval(u_canonical_J⁺, "Jp", gam), show_mesh=true, figure_size=figure_size, title="J⁺, t = $(round(tm, sigdigits=3))")
# save("./output/Jp_t$(tm).png", fig, px_per_unit=8)

# Initialize the solution for J⁻
u_riemann_J⁻ = initu(mesh, app, [riemann_initv, riemann_inits, riemann_initJ⁺, riemann_initJ⁻_wave])
u_canonical_J⁻ = similar(u_riemann_J⁻)

# Convert Riemann invariants to canonical variables for J⁻
for i in axes(u_riemann_J⁻, 1), k in axes(u_riemann_J⁻, 3)
    u_canonical_J⁻[i, :, k] .= riemann_to_canonical(u_riemann_J⁻[i, 1, k], u_riemann_J⁻[i, 2, k], u_riemann_J⁻[i, 3, k], u_riemann_J⁻[i, 4, k], gam)
end

# Time-stepping parameters for J⁻
Δt = 1e-4
nstep = 20
tm = 0.
time_total = 0.42
ncycl = Int(ceil(time_total / Δt / nstep))

# Plot initial condition for J⁻
fig = scaplot(mesh, eulereval(u_canonical_J⁻, "Jm", gam), show_mesh=true, figure_size=figure_size, title="J⁻, t = $(round(tm, sigdigits=3))")
# save("./output/Jm_initial.png", fig, px_per_unit=8)

# Time-stepping loop for J⁻
for i in 1:ncycl
    @info "$tm"
    rk4!(rinvexpl, master, mesh, app, u_canonical_J⁻, tm, Δt, nstep)
    tm += nstep * Δt
end

# Plot final condition for J⁻
fig = scaplot(mesh, eulereval(u_canonical_J⁻, "Jm", gam), show_mesh=true, figure_size=figure_size, title="J⁻, t = $(round(tm, sigdigits=3))")
# save("./output/Jm_t$(tm).png", fig, px_per_unit=8)

# Duct geometry with bump
db = 0.2
mesh_bump = mkmesh_square(m, n, porder, parity, nodetype)
mesh_bump = mkmesh_duct(mesh_bump, db, dt, H)

# Initial conditions for bump geometry
rho(x, y) = 1
ru(x, y) = rho(x, y) * ui[2] / ui[1]
rv(x, y) = rho(x, y) * ui[3] / ui[1]
rE(x, y) = 0.5 * (ru(x, y)^2 + rv(x, y)^2) / rho(x, y) + pinf / (gam - 1)
u = initu(mesh_bump, app, [rho, ru, rv, rE])

# Time-stepping parameters for bump geometry
Δt = 1e-3
nstep = 20
tm = 0.
time_total = 4
ncycl = Int(ceil(time_total / Δt / nstep))

# Time-stepping loop for bump geometry
for i in 1:ncycl
    @info "$tm"
    rk4!(rinvexpl, master, mesh_bump, app, u, tm, Δt, nstep)
    tm += nstep * Δt
end

# Plot results for bump geometry
fig = scaplot(mesh_bump, eulereval(u, "p", gam), show_mesh=true, figure_size=figure_size, title="pressure, t = $(round(tm, sigdigits=3))")
# save("./output/bump_pressure_t$(tm).png", fig, px_per_unit=8)
fig = scaplot(mesh_bump, eulereval(u, "M", gam), show_mesh=true, figure_size=figure_size, title="Mach number, t = $(round(tm, sigdigits=3))")
# save("./output/bump_Mach_t$(tm).png", fig, px_per_unit=8)

