using TwoDG
using BenchmarkTools
using LinearAlgebra
using CairoMakie
BLAS.set_num_threads(1)

porder = 4
m = n = 21
parity = 0
nodetype = 1

mesh = mkmesh_square(m, n, porder, parity, nodetype)
master = Master(mesh, 2 * porder)

kappa = 1
taud = 1
c = [10, 10]
restarts = [5, 10, 20, 40, 80, 160]
gmres_iters = zeros(length(restarts))
median_times = zeros(length(restarts))
min_times = zeros(length(restarts))
for (i, restart) in enumerate(restarts)
    @info "restart = $restart"
    param = Dict(:kappa => kappa, :c => c, :taud => taud)
    hdg_source(p) = 10 * ones(size(p, 1), 1)
    dbc(p) = zeros(size(p, 1), 1)

    uh, qh, uhath, gmres_iter = hdg_parsolve(master, mesh, hdg_source, dbc, param; restart)
    benchmark = @benchmark hdg_parsolve(master, mesh, $hdg_source, $dbc, $param; restart=$restart)
    gmres_iters[i] = gmres_iter
    median_times[i] = median(benchmark).time
    min_times[i] = minimum(benchmark).time
end
#%%
fig = Figure(size=(1200, 600))
ax_restart = Axis(fig[1, 1], title="GMRES Iterations vs Restart", xlabel="Restart", ylabel="GMRES Iterations", xscale=log2, yscale=log10)
ax_time = Axis(fig[1, 2], title="Median Time vs Restart", xlabel="Restart", ylabel="Time (s)", xscale=log2, yscale=log10)
scatterlines!(ax_restart, restarts, gmres_iters, markersize=10)
scatterlines!(ax_time, restarts, median_times ./ 1e9, markersize=10)
display(fig)
# save("output/hdg_cd_restart.png", fig, px_per_unit=4)
#%%
