using TwoDG
using Statistics
using CairoMakie

time_total = 2π
dt = 2π / 1000
nstep = 25
parity = 0
nodetype = 1
niter = Int(1000 / 25)

ngrids = [9, 13, 17]
ps = [2, 3, 4]

l2_errors = [zeros(length(ngrids)) for _ in ps]
l2_errors_distorted = [zeros(length(ngrids)) for _ in ps]

for (i, ngrid) in enumerate(ngrids), (j, p) in enumerate(ps)
    m = ngrid
    n = ngrid
    porder = p

    mesh = mkmesh_square(m, n, porder, parity, nodetype)
    master = Master(mesh, 2*porder)

    mesh_distorted = deepcopy(mesh)
    mkmesh_distort!(mesh_distorted)

    bcm = ones(Int64, 4)
    bcs = zeros(1, 1)

    vf(p) = hcat(-p[:, 2] .+ 0.5, p[:, 1] .- 0.5)

    app = mkapp_convection()
    app = App(app; bcm, bcs)
    app.arg[:vf] = vf

    init(x, y) = exp(-120 * ((x - 0.6)^2 + (y - 0.5)^2))

    u = initu(mesh, app, [init])
    @info "Computing MSE for p = $(p) and size = $(ngrids[i])"

    tm = 0.
    for i in 1:niter
        # println("Time: ", tm)
        
        # Update the solution using the convection operator
        # u .= rk4(rinvexpl, master, mesh, app, u, tm, dt, nstep)
        rk4!(rinvexpl, master, mesh, app, u, tm, dt, nstep)
        # fig = scaplot(mesh, u[:, 1, :], show_mesh=true, cmap=:viridis)
        # display(fig)
        tm += nstep * dt
    end
    l2_errors[j][i] = l2_error(mesh, u, init)

    @info "Computing MSE for distorted mesh p = $(p) and size = $(ngrids[i])"

    u_distorted = initu(mesh_distorted, app, [init])
    tm = 0.
    for i in 1:niter
        # println("Time: ", tm)
        
        # Update the solution using the convection operator
        # u .= rk4(rinvexpl, master, mesh, app, u, tm, dt, nstep)
        rk4!(rinvexpl, master, mesh_distorted, app, u_distorted, tm, dt, nstep)
        # fig = scaplot(mesh_distorted, u[:, 1, :], show_mesh=true, cmap=:viridis)
        # display(fig)
        tm += nstep * dt
    end
    l2_errors_distorted[j][i] = l2_error(mesh_distorted, u_distorted, init)
end
#%%
fig = Figure()
ax = Axis(fig[1, 1], xlabel="1/h", ylabel="L2 norm", title="Error convergence of DG explicit linear convection", yscale=log10, xscale=log2)
for i in eachindex(ngrids)
    scatterlines!(ax, ngrids .- 1, l2_errors[i], label="p=$(ps[i])")
end
axislegend(ax, position=:lb)
display(fig)
save("./output/dg_convection_convergence.png", fig, px_per_unit=8)
#%%
fig = Figure()
ax = Axis(fig[1, 1], xlabel="1/h", ylabel="L2 norm", title="Error convergence of DG explicit linear convection (distorted mesh)", yscale=log10, xscale=log2)
for i in eachindex(ngrids)
    scatterlines!(ax, ngrids .- 1, l2_errors_distorted[i], label="p=$(ps[i])")
end
axislegend(ax, position=:lb)
display(fig)
save("./output/dg_convection_convergence_distorted.png", fig, px_per_unit=8)
#%%