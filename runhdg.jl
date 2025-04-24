using TwoDG

porder = 4
ngauss = 2 * (porder + 1)
siz = 0.1
boundary_refinement = 3

mesh = mkmesh_circle(siz, porder, 1; boundary_refinement)
master = Master(mesh, ngauss)
meshplot_curved(mesh)

mesh1 = make_circle_nodes(mesh.p, mesh.t, porder + 1, 1)
master1 = Master(mesh1, ngauss)

meshplot_curved(mesh1)
#%%
kappa = 1
c = [100, 100]
taud = 1

param = Dict(:kappa => kappa, :c => c, :taud => taud)
hdg_source(p) = 10.0 .* ones(size(p, 1), 1)
dbc(p) = zeros(size(p, 1), 1)

# uh, qh, uhath = hdg_solve(master, mesh, hdg_source, dbc, param)
u, q, uh = hdg_solve(master, mesh, hdg_source, dbc, param)

fig = scaplot(mesh, u[:, 1, :], show_mesh=true)
save("output/hdg_circle_u.png", fig, px_per_unit=8)
# scaplot(mesh, -q[:, 2, :], show_mesh=true, title="q")
#%%
ustarh = hdg_postprocess(master, mesh, master1, mesh1, u, q ./ kappa)

scaplot(mesh1, ustarh[:, 1, :], show_mesh=true, title="u*")
scaplot(mesh, u[:, 1, :], show_mesh=true, title="u", limits=extrema(u[:, 1, :]))
#%%