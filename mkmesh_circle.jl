using TwoDG
using CSV
using DataFrames
using CairoMakie
using LinearAlgebra

p, t = make_circle_mesh(0.4)

p_unique, t_unique = fixmesh(p, t)

meshscatter(p_unique[:, 1], p_unique[:, 2])

f, t2f = mkt2f(t_unique)

boundary(p) = sqrt.(sum(p.^2, dims=2)) .> 1 - 2e-2

bndexpr = [boundary]

f = setbndnbrs(p_unique, f, bndexpr)

fcurved = f[:, 4] .< 0
tcurved = falses(size(t_unique, 1))
tcurved[f[fcurved, 3]] .= true

plocal, tlocal = uniformlocalpnts(3)

meshscatter(plocal[:, 2], plocal[:, 3])

mesh = TwoDG.Mesh(p_unique, t_unique, f, t2f, fcurved, tcurved, 3, plocal, tlocal)

#%%
fd_circle(p) = abs(sqrt(sum(p.^2)) - 1)
fds = [fd_circle]

mesh = createnodes(mesh, fds)

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
    # lines!(ax, dgnodes[tlocal[j, :], 1, i], dgnodes[tlocal[j, :], 2, i], color=:black)
    poly!(ax, dgnodes[tlocal[j, :], 1, i], dgnodes[tlocal[j, :], 2, i], strokecolor=:black, strokewidth=0.5)
end

t = mesh.t
for i in 1:size(t, 1)
    poly!(ax, mesh.p[t[i, :], 1], mesh.p[t[i, :], 2], strokewidth=2, strokecolor=:black, color=(:white, 0))
end

display(fig)

#%%