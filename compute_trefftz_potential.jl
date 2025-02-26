using TwoDG
using CairoMakie

V∞ = 1
α = 10
m = 20
n = 20
porder = 4
node_spacing_type = 1
tparam = [0.1, 0.05, 1.98]
np_foil = 120

xs, ys, chord = trefftz_points(tparam, np_foil)

mesh = mkmesh_trefftz(m, n, porder, node_spacing_type, tparam)

c = mesh.dgnodes[:, 1, :] .+ mesh.dgnodes[:, 2, :]
meshplot(mesh)
scaplot(mesh, c, show_mesh=true)
#%%
fig = Figure(size=(800, 800))
ax = Axis(fig[1, 1], aspect=1)
for i in 1:size(mesh.dgnodes, 3), j in 1:size(mesh.tlocal, 1)
    poly!(ax, mesh.dgnodes[mesh.tlocal[j, :], 1, i], mesh.dgnodes[mesh.tlocal[j, :], 2, i], strokecolor=:black, strokewidth=0.5)
end

t = mesh.t
for i in 1:size(t, 1)
    poly!(ax, mesh.p[t[i, :], 1], mesh.p[t[i, :], 2], strokewidth=2, strokecolor=:black, color=(:white, 0))
end

display(fig)
#%%
fig = Figure()
ax = Axis(fig[1, 1]; xlabel="x", ylabel="y")
scatter!(ax, xs, ys)
display(fig)
#%%
ψ, vx_analytical, vy_analytical, Γ_analytical = potential_trefftz(mesh.dgnodes[:, 1, :], mesh.dgnodes[:, 2, :], V=V∞, alpha=α, tparam=tparam)

scaplot(mesh, ψ, show_mesh=true)

#%%
trefftz(V∞, α, m, n, porder, node_spacing_type, tparam)
#%%
flat_dg = reshape(mesh.dgnodes, :, size(mesh.dgnodes, 2))

scatter(flat_dg[:, 1], flat_dg[:, 2])
scatter(vec(mesh.dgnodes[:, 1, :]), vec(mesh.dgnodes[:, 2, :]))

#%%
mesh, master,  ψ, vx, vy, vx_analytical, vy_analytical, Γ_analytical = trefftz(V∞, α, m, n, porder, node_spacing_type, tparam)
scaplot(mesh, ψ)
fig = scaplot(mesh, vx, show_mesh=true)
scaplot(mesh, vx_analytical, show_mesh=true)

scaplot(mesh, vy, show_mesh=true)
scaplot(mesh, vy_analytical, show_mesh=true)
#%%
vx_singularity_locations = findall(x -> abs(x) > 10, vx)
vy_singulatity_locations = findall(x -> abs(x) > 10, vy)
#%%
for i in vx_singularity_locations
    vx[i] = 0
end

for i in vy_singulatity_locations
    vy[i] = 0
end
#%%
fig = scaplot(mesh, vx, show_mesh=true)
# save("vx_trefftz.png", fig, px_per_unit=4)
scaplot(mesh, vx_analytical, show_mesh=true)

scaplot(mesh, vy, show_mesh=true)
scaplot(mesh, vy_analytical, show_mesh=true)
#%%