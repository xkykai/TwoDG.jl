using TwoDG
using CairoMakie

#%%
t = 10
porder = 3
name = "naca0012"
mesh = mkmesh_naca(t, porder, name)
#%%
fig = meshplot_curved(mesh, title="NACA 0012 mesh, t = $t, order = $porder", pplot=5)
save("./output/naca_mesh.png", fig, px_per_unit=8)
#%%