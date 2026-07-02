using TwoDG
using CairoMakie

m = 10
n = 8
parity = 1
porder = 2
H = 1
db = 0.2
dt = 0.1

mesh = mkmesh_square(m, n, porder, parity)
mesh = mkmesh_duct(mesh, db, dt, H)
#%%
fig = meshplot_curved(mesh, nodes=true, title="Duct mesh, m = $m, n = $n, parity = $parity, order = $porder, H = $H, db = $db, dt = $dt", pplot=5, figure_size=(800, 300))
save("./output/duct_mesh.png", fig, px_per_unit=8)
#%%