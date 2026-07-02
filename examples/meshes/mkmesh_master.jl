using TwoDG
using CairoMakie
using LinearAlgebra

porder = 3
type = 0

p = [0. 0; 1 0; 0 1]
t = [1 2 3]

f, t2f = mkt2f(t)
f[:, 4] .= -1 .* f[:, 1]

fcurved = falses(size(f, 1))
tcurved = falses(size(t, 1))

plocal, tlocal = uniformlocalpnts(porder)

mesh = TwoDG.Mesh(; p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal)
mesh = createnodes(mesh)

#%%
fig = Figure()
ax = Axis(fig[1, 1], aspect=1)
for i in 1:size(mesh.dgnodes, 3)
    scatter!(ax, mesh.dgnodes[:, 1, i], mesh.dgnodes[:, 2, i])
end
display(fig)
#%%
fig = Figure()
ax = Axis(fig[1, 1], aspect=1)
dgnodes = mesh.dgnodes
tlocal = mesh.tlocal
for i in 1:size(dgnodes, 3), j in 1:size(tlocal, 1)
    poly!(ax, dgnodes[tlocal[j, :], 1, i], dgnodes[tlocal[j, :], 2, i], strokecolor=:black, strokewidth=0.5)
end

t = mesh.t
for i in 1:size(t, 1)
    poly!(ax, mesh.p[t[i, :], 1], mesh.p[t[i, :], 2], strokewidth=2, strokecolor=:black, color=(:white, 0))
end

display(fig)

#%%
master = Master(mesh)

nfs1d = shape1d(master.porder, master.ploc1d, master.gp1d)

#%%
xs = 0:0.01:1
nfs1d = shape1d(master.porder, master.ploc1d, xs)
fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, xs, nfs1d[1, 1, :])
lines!(ax, xs, nfs1d[2, 1, :])
lines!(ax, xs, nfs1d[3, 1, :])
lines!(ax, xs, nfs1d[4, 1, :])
vlines!(ax, [1/3, 2/3], color=:black)
display(fig)
#%%
triangle_coords = uniformlocalpnts(100)[1][:, 2:end]

nfs2d = shape2d(master.porder, master.plocal, triangle_coords)
#%%
fig = Figure(size=(1500, 800))
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
ax3 = Axis(fig[1, 3])
ax4 = Axis(fig[1, 4])
ax5 = Axis(fig[1, 5])
ax6 = Axis(fig[2, 1])
ax7 = Axis(fig[2, 2])
ax8 = Axis(fig[2, 3])
ax9 = Axis(fig[2, 4])
ax10 = Axis(fig[2, 5])
heatmap!(ax1, triangle_coords[:, 1], triangle_coords[:, 2], nfs2d[1, 1, :], colormap=:turbo)
heatmap!(ax2, triangle_coords[:, 1], triangle_coords[:, 2], nfs2d[2, 1, :], colormap=:turbo)
heatmap!(ax3, triangle_coords[:, 1], triangle_coords[:, 2], nfs2d[3, 1, :], colormap=:turbo)
heatmap!(ax4, triangle_coords[:, 1], triangle_coords[:, 2], nfs2d[4, 1, :], colormap=:turbo)
heatmap!(ax5, triangle_coords[:, 1], triangle_coords[:, 2], nfs2d[5, 1, :], colormap=:turbo)
heatmap!(ax6, triangle_coords[:, 1], triangle_coords[:, 2], nfs2d[6, 1, :], colormap=:turbo)
heatmap!(ax7, triangle_coords[:, 1], triangle_coords[:, 2], nfs2d[7, 1, :], colormap=:turbo)
heatmap!(ax8, triangle_coords[:, 1], triangle_coords[:, 2], nfs2d[8, 1, :], colormap=:turbo)
heatmap!(ax9, triangle_coords[:, 1], triangle_coords[:, 2], nfs2d[9, 1, :], colormap=:turbo)
heatmap!(ax10, triangle_coords[:, 1], triangle_coords[:, 2], nfs2d[10, 1, :], colormap=:turbo)
display(fig)
#%%