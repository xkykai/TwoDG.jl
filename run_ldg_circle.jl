using TwoDG
using CairoMakie
using Statistics
using LinearAlgebra

nodetype = 1

kappa = 1
c11 = 10
c11int = 0

app = mkapp_convection_diffusion()
app.pg = true

function vf(p)
    return zero(p)
end

function src(ug, args...)
    return zero(ug) .+ 1
end

bcm = [1]
bcs = zeros(1, 1)
app = App(app; bcm, bcs, src)
app.arg[:vf] = vf
app.arg[:kappa] = kappa
app.arg[:c11] = c11
app.arg[:c11int] = c11int


time_total = 2
nstep = 128
dt = time_total / (nstep * 250)
ncycl = Int(ceil(time_total / dt / nstep))
#%%
size = 0.25
porder = 3
mesh = mkmesh_circle(size, porder, 1)
master = Master(mesh, 4*porder)

meshplot_curved(mesh)
u = initu(mesh, app, [0])

time = 0
for i in 1:ncycl
    @info "time = $(time)"
    rk4!(rldgexpl, master, mesh, app, u, time, dt, nstep)
    time += nstep * dt
    fig = scaplot(mesh, u[:, 1, :], show_mesh=true)
    display(fig)
end