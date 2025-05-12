using TwoDG
using BenchmarkTools
using LinearAlgebra
using CairoMakie
BLAS.set_num_threads(1)

porder = 5
m = n = 21
parity = 0
nodetype = 1

mesh = mkmesh_square(m, n, porder, parity, nodetype)
master = Master(mesh, 2 * porder)

kappa = 1
taud = 1
cs = [[1, 1], [10, 10], [40, 40]]
gmres_iters = zeros(length(cs))
for (i, c) in enumerate(cs)
    @info "c = $c"
    param = Dict(:kappa => kappa, :c => c, :taud => taud)
    hdg_source(p) = 10 * ones(size(p, 1), 1)
    dbc(p) = zeros(size(p, 1), 1)

    restart = 20
    uh, qh, uhath, gmres_iter = hdg_parsolve(master, mesh, hdg_source, dbc, param; restart)
    gmres_iters[i] = gmres_iter
    fig = scaplot(mesh, uh, show_mesh=true, title="u, c = $c")
    # save("./output/hdgpar_uhat_c_$(c[1])_$(c[2]).png", fig, px_per_unit=4)
end
#%%
uh, qh, uhath = hdg_solve(master, mesh, hdg_source, dbc, param)
scaplot(mesh, uh[:, 1, :], show_mesh=true, title="u")
#%%
