using TwoDG
using CSV
using DataFrames
using CairoMakie
using TwoDG.Meshes: transfinite_interpolation_triangle
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

dgnode_mesh = createnodes(mesh, fds)

#%%
fig = Figure()
ax = Axis(fig[1, 1], aspect=1)
for i in 1:size(dgnode_mesh.dgnodes, 3)
    scatter!(ax, dgnode_mesh.dgnodes[:, 1, i], dgnode_mesh.dgnodes[:, 2, i])
end
display(fig)
#%%