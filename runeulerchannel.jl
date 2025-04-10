using TwoDG
using CairoMakie

m = 16
n = 8
porder = 5

parity = 1
nodetype = 1
gam = 1.4
Minf = 0.3

figure_size = (1000, 400)

ui = [1.0, 1.0, 0, 0.5 + 1 / (gam * (gam - 1) * Minf^2)]

db = 0
dt = 0
H = 1
mesh = mkmesh_square(m, n, porder, parity, nodetype)
master = Master(mesh, 2*porder)
mesh = mkmesh_duct(mesh, db, dt, H)
mesh.fcurved .= false
mesh.tcurved .= false

bcm = [2, 1, 2, 1]
bcs = vcat(ui', 0.0 .* ui')
app = mkapp_euler()
app = App(app; bcm, bcs)
app.arg[:gamma] = gam

background_rho = 1
background_ru = 1 * ui[2] / ui[1]
background_rv = 1 * ui[3] / ui[1]
pinf = 1 / (gam * Minf^2)
background_rE = 0.5 * (background_ru^2 + background_rv^2) / background_rho + pinf / (gam - 1)
background_riemann = canonical_to_riemann(background_rho, background_ru, background_rv, background_rE, gam)

plane_wave(x, y) = x >= 1.0 && x <= 2.0 ? 0.1 * sin(2π * (x-1)) : 0

riemann_initv(x, y) = background_riemann[1]
riemann_inits(x, y) = background_riemann[2]
riemann_initJ⁺_wave(x, y) = background_riemann[3] + plane_wave(x, y)
riemann_initJ⁺(x, y) = background_riemann[3]
riemann_initJ⁻_wave(x, y) = background_riemann[4] + plane_wave(x, y)
riemann_initJ⁻(x, y) = background_riemann[4]

u_riemann_J⁺ = initu(mesh, app, [riemann_initv, riemann_inits, riemann_initJ⁺_wave, riemann_initJ⁻])
u_canonical_J⁺ = similar(u_riemann_J⁺)

for i in axes(u_riemann_J⁺, 1), k in axes(u_riemann_J⁺, 3)
    u_canonical_J⁺[i, :, k] .= riemann_to_canonical(u_riemann_J⁺[i, 1, k], u_riemann_J⁺[i, 2, k], u_riemann_J⁺[i, 3, k], u_riemann_J⁺[i, 4, k], gam)
end

Δt = 2e-4
nstep = 20
tm = 0.
time_total = 0.24
ncycl = Int(ceil(time_total / Δt / nstep))

fig = scaplot(mesh, eulereval(u_canonical_J⁺, "Jp", gam), show_mesh=true, figure_size=figure_size, title="J⁺, t = $(round(tm, sigdigits=3))")
# save("./output/Jp_initial.png", fig, px_per_unit=8)
for i in 1:ncycl
    @info "$tm"
    # Update the solution using the convection operator
    rk4!(rinvexpl, master, mesh, app, u_canonical_J⁺, tm, Δt, nstep)
    tm += nstep * Δt
end

fig = scaplot(mesh, eulereval(u_canonical_J⁺, "Jp", gam), show_mesh=true, figure_size=figure_size, title="J⁺, t = $(round(tm, sigdigits=3))")
# save("./output/Jp_t$(tm).png", fig, px_per_unit=8)
#%%
u_riemann_J⁻ = initu(mesh, app, [riemann_initv, riemann_inits, riemann_initJ⁺, riemann_initJ⁻_wave])
u_canonical_J⁻ = similar(u_riemann_J⁻)

for i in axes(u_riemann_J⁻, 1), k in axes(u_riemann_J⁻, 3)
    u_canonical_J⁻[i, :, k] .= riemann_to_canonical(u_riemann_J⁻[i, 1, k], u_riemann_J⁻[i, 2, k], u_riemann_J⁻[i, 3, k], u_riemann_J⁻[i, 4, k], gam)
end

Δt = 1e-4
nstep = 20
tm = 0.
time_total = 0.42
ncycl = Int(ceil(time_total / Δt / nstep))

fig = scaplot(mesh, eulereval(u_canonical_J⁻, "Jm", gam), show_mesh=true, figure_size=figure_size, title="J⁻, t = $(round(tm, sigdigits=3))")
# save("./output/Jm_initial.png", fig, px_per_unit=8)
for i in 1:ncycl
    @info "$tm"
    # Update the solution using the convection operator
    rk4!(rinvexpl, master, mesh, app, u_canonical_J⁻, tm, Δt, nstep)
    tm += nstep * Δt
end
fig = scaplot(mesh, eulereval(u_canonical_J⁻, "Jm", gam), show_mesh=true, figure_size=figure_size, title="J⁻, t = $(round(tm, sigdigits=3))")
# save("./output/Jm_t$(tm).png", fig, px_per_unit=8)
#%%
db = 0.2
mesh_bump = mkmesh_square(m, n, porder, parity, nodetype)
mesh_bump = mkmesh_duct(mesh_bump, db, dt, H)

vv(dg) = ones(size(dg, 1), size(dg, 3)) .* ui[3]

rho(x, y) = 1
ru(x, y) = rho(x, y) * ui[2] / ui[1]
rv(x, y) = rho(x, y) * ui[3] / ui[1]
rE(x, y) = 0.5 * (ru(x, y)^2 + rv(x, y)^2) / rho(x, y) + pinf / (gam - 1)
u = initu(mesh_bump, app, [rho, ru, rv, rE])

Δt = 1e-3
nstep = 20
tm = 0.
time_total = 4
ncycl = Int(ceil(time_total / Δt / nstep))
tm = 0.
for i in 1:ncycl
    @info "$tm"
    # Update the solution using the convection operator
    rk4!(rinvexpl, master, mesh_bump, app, u, tm, Δt, nstep)
    tm += nstep * Δt
end
#%%
fig = scaplot(mesh_bump, eulereval(u, "p", gam), show_mesh=true, figure_size=figure_size, title="pressure, t = $(round(tm, sigdigits=3))")
# save("./output/bump_pressure_t$(tm).png", fig, px_per_unit=8)
fig = scaplot(mesh_bump, eulereval(u, "M", gam), show_mesh=true, figure_size=figure_size, title="Mach number, t = $(round(tm, sigdigits=3))")
# save("./output/bump_Mach_t$(tm).png", fig, px_per_unit=8)
#%%

