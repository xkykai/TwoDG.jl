using TwoDG
using CairoMakie
using Statistics
using Symbolics
using LinearAlgebra

nodetype = 1

κ = 0.01
c = [1, -2]
s = 1 
param = (; κ, c, s)

@variables x y
u = cos(π/2 * sqrt(x^2 + y^2))

exact_expr = build_function(u, x, y)
exact = eval(exact_expr)

∂x = Differential(x)
∂y = Differential(y)
∂x² = ∂x * ∂x
∂y² = ∂y * ∂y

∇²u = expand_derivatives(∂x²(u) + ∂y²(u))
∇u = expand_derivatives.([∂x(u), ∂y(u)])

f = -κ * ∇²u + c ⋅ ∇u + s * u
source_expr = build_function(f, x, y)
source = eval(source_expr)
#%%
one_over_sizes = [2, 3, 5, 6, 8, 16]
sizes = 1 ./ one_over_sizes
ps = [2, 3, 4]
L2_errors = [zeros(length(sizes)) for _ in ps]

for i in eachindex(ps), j in eachindex(sizes)
    @info "Computing MSE for p = $(ps[i]) and size = $(sizes[j])"
    p = ps[i]
    size = sizes[j]
    mesh = mkmesh_circle(size, p, 1)
    master = Master(mesh, 4*p)
    uh, energy = cg_solve(mesh, master, source, param)
    uexact = exact.(mesh.pcg[:, 1], mesh.pcg[:, 2])
    L2_errors[i][j] = √(l2_error(mesh, uh, exact))
end

#%%
fig = Figure()
ax = Axis(fig[1, 1], xlabel="1/h", ylabel="L2 norm", title="Error convergence of continuous Galerkin, circular domain", yscale=log10, xscale=log10)
for i in eachindex(ps)
    scatterlines!(ax, one_over_sizes, L2_errors[i], label="p = $(ps[i])")
end
axislegend(ax, position=:lb)
display(fig)
# save("./output/cg_circle_convergence.pdf", fig)
save("./output/cg_circle_convergence.png", fig, px_per_unit=8)
#%%