using TwoDG
using CairoMakie
using LinearAlgebra
using Statistics

parity = 0
nodetype = 1

κ = 0.01
c = [1, -2]
s = 1 
param = (; κ, c, s)

n = 3
m = 3
exact(x, y) = sin(n * π * x) * sin(m * π * y)
function source(x, y; n, m, κ, c, s)
    return κ * (n^2 + m^2) * π^2 * (sin(n * π * x) * sin(m * π * y)) + 
           c[1] * n * π * (cos(n * π * x) * sin(m * π * y)) + 
           c[2] * m * π * (sin(n * π * x) * cos(m * π * y)) + 
           s * (sin(n * π * x) * sin(m * π * y))
end

source(x, y) = source(x, y, n=n, m=m, κ=κ, c=c, s=s)

ngrids = [5, 9, 17, 33]
ps = [1, 2, 3, 4]
L2_errors = [zeros(length(ngrids)) for _ in ps]

for i in eachindex(ps), j in eachindex(ngrids)
    @info "Computing MSE for p = $(ps[i]) and ngrid = $(ngrids[j])"
    p = ps[i]
    ngrid = ngrids[j]
    mesh = mkmesh_square(ngrid, ngrid, p, parity, nodetype)
    master = Master(mesh, 4*p)
    uh, energy, u = cg_solve(mesh, master, source, param)
    uexact = exact.(mesh.pcg[:, 1], mesh.pcg[:, 2])
    L2_errors[i][j] = l2_error(mesh, uh, exact)
end
#%%
fig = Figure()
ax = Axis(fig[1, 1], xlabel="1/h", ylabel="L2 norm", title="Error convergence of continuous Galerkin", yscale=log10, xscale=log10)
for i in eachindex(ps)
    scatterlines!(ax, ngrids .- 1, L2_errors[i], label="p = $(ps[i])")
end
axislegend(ax, position=:lb)
display(fig)
save("./output/cg_square_convergence.pdf", fig)
#%%