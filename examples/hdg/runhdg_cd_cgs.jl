using TwoDG
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
preconditioner = false

# Modified Gram-Schmidt
u, q, uh, gmres_iter = hdg_parsolve(master, mesh, hdg_source, dbc, param; restart, ortho=1, preconditioner)
fig = scaplot(mesh, u, show_mesh=true, title="u, Modified Gram-Schmidt")
# save("output/hdg_cd_mgs.png", fig, px_per_unit=4)

# Classical Gram-Schmidt
u_c, q_c, uh_c, gmres_iter_c = hdg_parsolve(master, mesh, hdg_source, dbc, param; restart, ortho=0, preconditioner)
fig = scaplot(mesh, u_c, show_mesh=true, title="u, Classical Gram-Schmidt")
# save("output/hdg_cd_cgs.png", fig, px_per_unit=4)
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
preconditioner = false
#%%
# Modified Gram-Schmidt
@info "Running Modified Gram-Schmidt"
@benchmark hdg_parsolve(master, mesh, hdg_source, dbc, param; restart, ortho=1, preconditioner)

# Classical Gram-Schmidt
@info "Running Classical Gram-Schmidt"
@benchmark hdg_parsolve(master, mesh, hdg_source, dbc, param; restart, ortho=0, preconditioner)
#%%