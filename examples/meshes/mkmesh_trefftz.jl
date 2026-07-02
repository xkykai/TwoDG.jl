using TwoDG
using CairoMakie

m = 15
n = 30
node_spacing_type = 0
tparam = [0.1, 0.05, 1.98]
porder = 3

mesh = mkmesh_trefftz(m, n, porder, node_spacing_type, tparam)
#%%
fig = meshplot_curved(mesh, title="Trefftz mesh, m = $m, n = $n, parity = $parity, order = $porder, tparam = $tparam", pplot=5)
save("./output/trefftz_mesh.png", fig, px_per_unit=8)
#%%

