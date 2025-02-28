using TwoDG
using CairoMakie

m = 10
n = 10
parity = 0

mesh = mkmesh_square(m, n, porder, parity)
#%%
fig = meshplot_curved(mesh, nodes=true, title="Square mesh, m = $m, n = $n, parity = $parity, order = $porder", pplot=5, figure_size=(800, 800))
save("./output/square_mesh.png", fig, px_per_unit=8)
#%%
