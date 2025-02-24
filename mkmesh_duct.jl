using TwoDG
using CSV
using DataFrames
using CairoMakie
using LinearAlgebra

m = 10
n = 8
parity = 1
porder = 2

p, t = make_square_mesh(m, n, parity)

p_unique, t_unique = fixmesh(p, t)

scatter(p_unique[:, 1], p_unique[:, 2])

f, t2f = mkt2f(t_unique)

boundary_ϵ = 1e-3
boundary_left(p) = (p[:, 1]) .< boundary_ϵ
boundary_right(p) = (p[:, 1]) .> 1 - boundary_ϵ
boundary_bottom(p) = (p[:, 2]) .< boundary_ϵ
boundary_top(p) = (p[:, 2]) .> 1 - boundary_ϵ

bndexpr = [boundary_left, boundary_right, boundary_bottom, boundary_top]

f = setbndnbrs(p_unique, f, bndexpr)

fcurved = falses(size(f, 1))
tcurved = falses(size(t_unique, 1))
tcurved[f[fcurved, 3]] .= true

plocal, tlocal = uniformlocalpnts(porder)

mesh = TwoDG.Mesh(p_unique, t_unique, f, t2f, fcurved, tcurved, porder, plocal, tlocal)

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
H = 1
db = 0.2
dt = 0.1

p = mesh.p
mesh.fcurved .= true
mesh.tcurved .= true

pnew = zeros(size(p))
pnew[:,1] .= 3.0 .* p[:,1]

ii = findall(x -> x <= 1 || x >= 2, pnew[:, 1])
pnew[ii, 2] .= H .* p[ii, 2]

ii = findall(x -> x > 1 && x < 2, pnew[:, 1])
pnew[ii, 2] .= p[ii, 2] .* (H .- dt .* sin.(π .* (pnew[ii, 1] .- 1)).^2) .+ 
                (1 .- p[ii, 2]) .* (db .* sin.(π .* (pnew[ii, 1] .- 1)).^2)

scatter(pnew[:, 1], pnew[:, 2])
#%%
new_mesh = TwoDG.Mesh(pnew, mesh.t, mesh.f, mesh.t2f, mesh.fcurved, mesh.tcurved, mesh.porder, mesh.plocal, mesh.tlocal)

fd_left(p) = abs(p[1])
fd_right(p) = abs(p[1] .- 3)

function fd_bottom(p, H, db)
    x, y = p
    if x <= 1 || x >= 2
        return abs(y)
    else
        return abs(y - ((1 - y/H) .* (db * sin(π * (x - 1))^2)))
    end
end
fd_bottom(p) = fd_bottom(p, H, db)

xs = 0:0.01:3
ys = 0:0.01:1
fdb = zeros(length(xs), length(ys))
for i in 1:length(xs), j in 1:length(ys)
    fdb[i, j] = fd_bottom([xs[i], ys[j]])
end
heatmap(xs, ys, fdb)

function fd_top(p, H, dt)
    x, y = p
    if x <= 1 || x >= 2
        return abs(y - H)
    else
        return abs(y - (H - dt * sin(π * (x - 1))^2))
    end
end
fd_top(p) = fd_top(p, H, dt)

fdt = zeros(length(xs), length(ys))
for i in 1:length(xs), j in 1:length(ys)
    fdt[i, j] = fd_top([xs[i], ys[j]])
end

heatmap(xs, ys, fdt)

fds = [fd_left, fd_right, fd_bottom, fd_top]

new_mesh_dg = createnodes(new_mesh, fds)
#%%
fig = Figure()
ax = Axis(fig[1, 1], aspect=1)
for i in 1:size(new_mesh_dg.dgnodes, 3)
    scatter!(ax, new_mesh_dg.dgnodes[:, 1, i], new_mesh_dg.dgnodes[:, 2, i])
end
display(fig)
#%%
fig = Figure()
ax = Axis(fig[1, 1], aspect=DataAspect())
dgnodes = new_mesh_dg.dgnodes
tlocal = new_mesh_dg.tlocal
t = new_mesh_dg.t

for i in 1:size(dgnodes, 3), j in 1:size(tlocal, 1)
    poly!(ax, dgnodes[tlocal[j, :], 1, i], dgnodes[tlocal[j, :], 2, i], strokecolor=:black, strokewidth=0.5)
end

for i in 1:size(t, 1)
    poly!(ax, new_mesh_dg.p[t[i, :], 1], new_mesh_dg.p[t[i, :], 2], strokewidth=2, strokecolor=:black, color=(:white, 0))
end

display(fig)
#%%
