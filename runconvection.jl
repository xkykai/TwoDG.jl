using TwoDG
using Statistics

m = 10
n = 10
porder = 3

time_total = 2π
dt = 2π / 600
nstep = 25
parity = 0
nodetype = 1
niter = Int(600 / 25)

mesh = mkmesh_square(m, n, porder, parity, nodetype)
master = Master(mesh, 2*porder)

mesh2 = deepcopy(mesh)
master2 = deepcopy(master)

bcm = ones(Int64, 4)
bcs = zeros(1, 1)

vf(p) = hcat(-p[:, 2] .+ 0.5, p[:, 1] .- 0.5)

app = mkapp_convection()
app = App(app; bcm, bcs)
app.arg[:vf] = vf

init(x, y) = exp(-120 * ((x - 0.6)^2 + (y - 0.5)^2))

u = initu(mesh, app, [init])

scaplot(mesh, u[:, 1, :], show_mesh=true)


a = rinvexpl(master, mesh, app, u, 0)
scaplot(mesh, a[:, 1, :], show_mesh=true, cmap=:viridis)
# #%%
tm = 0.

for i in 1:niter
    tm += nstep * dt
    # println("Time: ", tm)
    
    # Update the solution using the convection operator
    # u .= rk4(rinvexpl, master, mesh, app, u, tm, dt, nstep)
    rk4!(rinvexpl, master, mesh, app, u, tm, dt, nstep)
    fig = scaplot(mesh, u[:, 1, :], show_mesh=true, cmap=:viridis)
    display(fig)
end
#%%
mkmesh_distort!(mesh2)
meshplot_curved(mesh2, pplot=6)
#%%
a = rinvexpl(master, mesh, app, u, 0)
scaplot(mesh, a[:, 1, :], show_mesh=true, cmap=:viridis)
#%%
a2 = rinvexpl(master2, mesh2, app, u, 0)
scaplot(mesh2, a2[:, 1, :], show_mesh=true, cmap=:viridis)
#%%
tm = 0.
u2 = initu(mesh2, app, [init])

for i in 1:niter
    tm += nstep * dt
    println("Time: ", tm)
    
    # Update the solution using the convection operator
    # u .= rk4(rinvexpl, master, mesh, app, u, tm, dt, nstep)
    rk4!(rinvexpl, master2, mesh2, app, u2, tm, dt, nstep)
    fig = scaplot(mesh2, u2[:, 1, :], show_mesh=true, cmap=:viridis)
    display(fig)
end
#%%
mean((u2 .- initu(mesh2, app, [init])).^2)
#%%
