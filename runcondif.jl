using TwoDG
using CairoMakie

m = 21
n = 11
porder = 4
parity = 0
kappa = 0.01

time_total = 2
nstep = 5
dt = time_total / (nstep * 250)
ncycl = Int(ceil(time_total / dt / nstep))
nper = 5

c11 = 10
c11int = 0

mesh = mkmesh_square(m, n, porder, parity, 1)
mesh.p[:, 1] = 10.0 .* mesh.p[:, 1]
mesh.p[:, 2] = 2.5 .* mesh.p[:, 2] .- 1.25
mesh.dgnodes[:, 1, :] .= 10.0 .* mesh.dgnodes[:, 1, :]
mesh.dgnodes[:, 2, :] .= 2.5 .* mesh.dgnodes[:, 2, :] .- 1.25

meshplot_curved(mesh)

master = Master(mesh, 3*porder)
Re = 100
g = 0.5*Re - sqrt(0.25*Re^2 + 4.0*π^2)

# Define the Kovasznay flow functions
function xf(p)
    return 1.0 .- exp.(g .* p[:,1]) .* cos.(2.0 .* π .* p[:,2])
end

function yf(p)
    return 0.5 .* g .* exp.(g .* p[:,1]) .* sin.(2.0 .* π .* p[:,2]) ./ π
end

function vf(p)
    return hcat(xf(p), yf(p))
end

app = mkapp_convection_diffusion()
app.pg = true

bcm = [2, 2, 2, 1]
bcs = zeros(2, 1)

app = App(app; bcm, bcs)
app.arg[:vf] = vf
app.arg[:kappa] = kappa
app.arg[:c11] = c11
app.arg[:c11int] = c11int

function gaus(x, y, s)
    return exp.(-((x .- 1.0).^2 .+ (y .- s).^2) ./ 0.25)
end

function init(x, y)
    return gaus(x, y, 0.0) .+ gaus(x, y, 0.5) .+ gaus(x, y, -0.5)
end

elloc = findfirst(i -> sum((mesh.dgnodes[:, 1, i] .≈ 4) .& (mesh.dgnodes[:, 2, i] .≈ 0)) > 0, axes(mesh.dgnodes, 3))
iploc = findfirst(i -> mesh.dgnodes[i, 1, elloc] ≈ 4 && mesh.dgnodes[i, 2, elloc] ≈ 0, axes(mesh.dgnodes, 1))

u_locs = zeros(ncycl * nper)

#%%
time = 0
u = initu(mesh, app, [0])
u .+= initu(mesh, app, [init])
scaplot(mesh, u[:, 1, :], show_mesh=true, figure_size=(800, 300), title="t = $(round(time, sigdigits=3))")

u_t6 = zero(u)

i_uloc = 1
for iper in 1:nper
    fig = scaplot(mesh, u[:, 1, :], show_mesh=true, figure_size=(800, 300), title="t = $(round(time, sigdigits=3))")
    display(fig)

    if iper == 3
        u_t6 .= u
    end

    u .+= initu(mesh, app, [init])
    for i in 1:ncycl
        @info "$time"
        rk4!(rldgexpl, master, mesh, app, u, time, dt, nstep)
        # rk4!(rinvexpl, master, mesh, app, u, time, dt, nstep)
        time += nstep * dt

        u_locs[i_uloc] = u[iploc, 1, elloc]
        i_uloc += 1

    end
end
#%%
