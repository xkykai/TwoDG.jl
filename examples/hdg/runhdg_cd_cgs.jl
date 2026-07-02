# NOTE: this example originally compared classical vs modified Gram-Schmidt
# orthogonalization in the hand-rolled HDG GMRES. That solver has been
# replaced by Krylov.jl (`hdg_parsolve` now wraps `hdg_gmres_ka`), so the
# comparison here is preconditioned vs unpreconditioned GMRES instead.
using TwoDG
using BenchmarkTools
using LinearAlgebra
using CairoMakie
BLAS.set_num_threads(1)

porder = 10
ngauss = 2 * (porder + 1)
boundary_refinement = 3
siz = 0.1

hdg_source(p) = 10 .* ones(size(p, 1), 1)
dbc(p) = zeros(size(p, 1), 1)

mesh = mkmesh_circle(siz, porder, 1; boundary_refinement)
master = Master(mesh, ngauss)

kappa = 1e-6
taud = 1
c = [10000, 10000]
restart = 800
param = Dict(:kappa => kappa, :c => c, :taud => taud)

# Block-Jacobi preconditioned
u, q, uh, gmres_iter = hdg_parsolve(master, mesh, hdg_source, dbc, param; restart)
fig = scaplot(mesh, u[:, 1, :], show_mesh=true, title="u, preconditioned ($gmres_iter its)")
# save("output/hdg_cd_precond.png", fig, px_per_unit=4)

# Unpreconditioned
u_c, q_c, uh_c, gmres_iter_c = hdg_parsolve(master, mesh, hdg_source, dbc, param;
                                            restart, preconditioner=false)
fig = scaplot(mesh, u_c[:, 1, :], show_mesh=true, title="u, unpreconditioned ($gmres_iter_c its)")
# save("output/hdg_cd_noprecond.png", fig, px_per_unit=4)
#%%
porder = 5
ngauss = 2 * porder
boundary_refinement = 3
siz = 0.1

hdg_source(p) = 10 .* ones(size(p, 1), 1)
dbc(p) = zeros(size(p, 1), 1)

mesh = mkmesh_circle(siz, porder, 1; boundary_refinement)
master = Master(mesh, ngauss)
#%%
kappa = 1
taud = 1
c = [1000, 1000]
restart = 100
param = Dict(:kappa => kappa, :c => c, :taud => taud)
#%%
@info "Preconditioned"
@benchmark hdg_parsolve(master, mesh, hdg_source, dbc, param; restart)

@info "Unpreconditioned"
@benchmark hdg_parsolve(master, mesh, hdg_source, dbc, param; restart, preconditioner=false)
#%%
