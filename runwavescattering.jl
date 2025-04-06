using TwoDG
using Statistics

m = 11
n = 20
porder = 4

time_total = 100
dt = 0.6e-2
nstep = 20
nodetype = 1
ncycl = Int(ceil(time_total / (nstep * dt)))

c = 1
k = [3, 0]
kmod = sqrt(sum(k.^2))

tparam = [0, 0, 1]
mesh = mkmesh_trefftz(m, n, porder, nodetype, tparam)
master = Master(mesh, 2*porder)

meshplot_curved(mesh)

bcm = [2, 3]
bcs = zeros(3, 3)
app = mkapp_wave()
app = App(app; bcm, bcs)

ub(c, k, p, time) = sin.(p[:,1] .* k[1] .+ p[:,2] .* k[2] .- c * sqrt(k[1]^2 + k[2]^2) * time)

app.pg = true
app.arg[:c] = c
app.arg[:k] = k
app.arg[:f] = ub

u = initu(mesh, app, [0, 0, 0])
ue = zeros(size(u))

a = rinvexpl(master, mesh, app, u, 0)
scaplot(mesh, a[:, 1, :], show_mesh=true, cmap=:viridis)
#%%
tm = 0.

for i in 1:ncycl
    # Update `ue` based on the current time `tm`
    ue[:, 3, :] .= sin.(mesh.dgnodes[:, 1, :] .* k[1] .+ mesh.dgnodes[:, 2, :] .* k[2] .- c * kmod * tm)
    ue[:, 1, :] .= -k[1] * ue[:, 3, :] / kmod
    ue[:, 2, :] .= -k[2] * ue[:, 3, :] / kmod

    # Set initial condition for `u` if it's the first iteration
    if i == 1
        u .= ue
    end

    # Plot the current solution
    fig = scaplot(mesh, u[:, 3, :], show_mesh=true)
    display(fig)

    # Update the solution using the convection operator
    rk4!(rinvexpl, master, mesh, app, u, tm, dt, nstep)

    # Increment time
    tm += nstep * dt
end
#%%