using TwoDG
using CairoMakie

siz = 0.4
porder = 3
mesh = mkmesh_circle(siz, porder)

fig = meshplot_curved(mesh, nodes=true, title="Circle mesh, size = $(siz), order = $(porder)", pplot=5)
save("./output/circle_mesh.png", fig, px_per_unit=8)
#%%