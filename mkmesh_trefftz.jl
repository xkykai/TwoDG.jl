using TwoDG
using CSV
using DataFrames
using CairoMakie
using LinearAlgebra

m = 15
n = 30
node_spacing_type = 0
tparam = [0.1, 0.05, 1.98]
porder = 3

n = 2 * Int(ceil(n / 2))

p0, t0 = make_square_mesh(m, n ÷ 2, 0)
p1, t1 = make_square_mesh(m, n ÷ 2, 1)

nump = size(p0, 1)

t1 = t1 .+ nump
p1[:, 2] .+= 1

p = vcat(p0, p1)
t = vcat(t0, t1)

scatter(p[:, 1], p[:, 2])

plocal, tlocal = uniformlocalpnts(3)

mesh = TwoDG.Mesh(p, t, 3, plocal, tlocal)

mesh = createnodes(mesh)

mesh.p[:, 1] .*= 2
mesh.p[:, 2] .*= π

z = mesh.p[:, 1] .+ im * mesh.p[:, 2]
w = exp.(z)

mesh.p[:, 1] = real(w)
mesh.p[:, 2] = imag(w)

p_unique, t_unique = fixmesh(mesh.p, mesh.t)

mesh = TwoDG.Mesh(p_unique, t_unique, nothing, nothing, nothing, nothing, porder, plocal, tlocal, mesh.dgnodes)

f, t2f = mkt2f(t_unique)

fcurved = trues(size(f, 1))
tcurved = trues(size(t_unique, 1))

boundary1(p) = vec(sqrt.(sum(p.^2, dims=2)) .< 2)
boundary2(p) = vec(sqrt.(sum(p.^2, dims=2)) .> 2)

bndexpr = [boundary1, boundary2]

f = setbndnbrs(p_unique, f, bndexpr)

mesh = TwoDG.Mesh(p_unique, t_unique, f, t2f, fcurved, tcurved, porder, plocal, tlocal, mesh.dgnodes)

#%%
# now do a K-T transformation
x0 = tparam[1]
y0 = tparam[2]
n = tparam[3]

rot = atan(y0, 1+x0)
r = sqrt((1+x0)^2 + y0^2)

w = mesh.p[:, 1] .+ im * mesh.p[:, 2]
w .= r * exp(-im * rot) .* w .- x0 .+ im * y0

z = ((w .- 1) ./ (w .+ 1)).^n
w .= ((1 .+ z) ./ (1 .- z)) .* n

mesh.p[:, 1] = real(w)
mesh.p[:, 2] = imag(w)

z = 2 .* mesh.dgnodes[:, 1, :] .- im * π .* mesh.dgnodes[:, 2, :]
w = exp.(z)
w .= r * exp(-im * rot) .* w .- x0 .+ im * y0

z = ((w .- 1) ./ (w .+ 1)).^n
w .= ((1 .+ z) ./ (1 .- z)) .* n

mesh.dgnodes[:, 1, :] = real(w)
mesh.dgnodes[:, 2, :] = imag(w)

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