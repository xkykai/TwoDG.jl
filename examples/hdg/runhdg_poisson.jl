using TwoDG
using Symbolics
using LinearAlgebra
using CairoMakie
using Makie

#%%
@variables x y
u = sin(2π*x) * sin(2π*y)
exact_expr = build_function(u, x, y)
exact = eval(exact_expr)

∂x = Differential(x)
∂y = Differential(y)
∂x² = ∂x * ∂x
∂y² = ∂y * ∂y

∇²u = expand_derivatives(∂x²(u) + ∂y²(u))
∇u = expand_derivatives.([∂x(u), ∂y(u)])
∂xu = expand_derivatives(∂x(u))
∂yu = expand_derivatives(∂y(u))

∂u∂x_expr = build_function(∂xu, x, y)
∂u∂y_expr = build_function(∂yu, x, y)

qx_exact = eval(∂u∂x_expr)
qy_exact = eval(∂u∂y_expr)

kappa = 1
c = [0, 0]

f = -kappa * ∇²u + c ⋅ ∇u
rhs_expr = build_function(f, x, y)
rhs = eval(rhs_expr)

hdg_source(p) = rhs.(p[:, 1], p[:, 2])
dbc(p) = zeros(size(p, 1), 1)
#%%
tauds_type = [1, 2, 3]
ms = [9, 17, 33]
porders = [1, 2, 3]

l2_errors = [[zeros(length(ms)) for _ in porders] for _ in tauds_type]
l2_errors_postprocess = [[zeros(length(ms)) for _ in porders] for _ in tauds_type]

l2_errors_qx = [[zeros(length(ms)) for _ in porders] for _ in tauds_type]
l2_errors_qy = [[zeros(length(ms)) for _ in porders] for _ in tauds_type]

for i in eachindex(tauds_type), j in eachindex(ms), k in eachindex(porders)
    m = ms[j]
    h = 1 / (m - 1)
    if tauds_type[i] == 1
        taud = h
    elseif tauds_type[i] == 2
        taud = 1
    elseif tauds_type[i] == 3
        taud = 1 / h
    end

    porder = porders[k]
    ngauss = 3 * (porder + 1)

    @info "Computing for p = $(porder), m = $(m), and τd = $(taud)"
    
    mesh = mkmesh_square(m, m, porder, 0, 1)
    master = Master(mesh, ngauss)
    
    mesh1 = mkmesh_square(m, m, porder + 1, 0, 1)
    master1 = Master(mesh1, ngauss)
    
    param = Dict(:kappa => kappa, :c => c, :taud => taud)
    
    u, q, uh = hdg_solve(master, mesh, hdg_source, dbc, param)
    ustarh = hdg_postprocess(master, mesh, master1, mesh1, u, q ./ kappa)

    l2_errors[i][k][j] = l2_error(mesh, u, exact)
    l2_errors_postprocess[i][k][j] = l2_error(mesh1, ustarh, exact)

    l2_errors_qx[i][k][j] = l2_error(mesh, -q[:, 1, :], qx_exact)
    l2_errors_qy[i][k][j] = l2_error(mesh, -q[:, 2, :], qy_exact)

    fig = scaplot(mesh1, ustarh[:, 1, :], show_mesh=true, title="u*")
    display(fig)

    if j == length(ms) && k == length(porders) && taud == 1
        # save("./output/uh_star_$(tauds_type[i]).png", fig, px_per_unit=8)
    end
end
#%%
fig = Figure(size = (1200, 1200))
ax1 = Axis(fig[1, 1], xlabel="1/h", ylabel="L2 norm", title="τd = h", yscale=log10, xscale=log2)
ax2 = Axis(fig[2, 1], xlabel="1/h", ylabel="L2 norm", title="τd = 1", yscale=log10, xscale=log2)
ax3 = Axis(fig[3, 1], xlabel="1/h", ylabel="L2 norm", title="τd = 1/h", yscale=log10, xscale=log2)
axqx1 = Axis(fig[1, 2], xlabel="1/h", ylabel="L2 norm", title="τd = h", yscale=log10, xscale=log2)
axqx2 = Axis(fig[2, 2], xlabel="1/h", ylabel="L2 norm", title="τd = 1", yscale=log10, xscale=log2)
axqx3 = Axis(fig[3, 2], xlabel="1/h", ylabel="L2 norm", title="τd = 1/h", yscale=log10, xscale=log2)
axqy1 = Axis(fig[1, 3], xlabel="1/h", ylabel="L2 norm", title="τd = h", yscale=log10, xscale=log2)
axqy2 = Axis(fig[2, 3], xlabel="1/h", ylabel="L2 norm", title="τd = 1", yscale=log10, xscale=log2)
axqy3 = Axis(fig[3, 3], xlabel="1/h", ylabel="L2 norm", title="τd = 1/h", yscale=log10, xscale=log2)

Label(fig[0, 1], "L2 norm of u", tellwidth=false)
Label(fig[0, 2], "L2 norm of qx", tellwidth=false)
Label(fig[0, 3], "L2 norm of qy", tellwidth=false)

axs = [ax1, ax2, ax3]
axqxs = [axqx1, axqx2, axqx3]
axqys = [axqy1, axqy2, axqy3]
colors = Makie.wong_colors()
linewidth = 3
for i in eachindex(tauds_type)
    for j in eachindex(porders)
        scatterlines!(axs[i], ms .- 1, l2_errors[i][j], label="p = $(porders[j])", linewidth=linewidth, color=colors[j])
    end
    for j in eachindex(porders)
        scatterlines!(axs[i], ms .- 1, l2_errors_postprocess[i][j], label="p = $(porders[j]), postprocessed", linewidth=linewidth, linestyle=:dash, color=colors[j])
    end

    for j in eachindex(porders)
        scatterlines!(axqxs[i], ms .- 1, l2_errors_qx[i][j], label="p = $(porders[j])", linewidth=linewidth, color=colors[j])
    end
    for j in eachindex(porders)
        scatterlines!(axqys[i], ms .- 1, l2_errors_qy[i][j], label="p = $(porders[j])", linewidth=linewidth, color=colors[j])
    end
end
Legend(fig[4, 1:3], axs[1], orientation=:horizontal, tellwidth=false, patchsize=(40, 20))
display(fig)
# save("./output/hdg_convergence.png", fig, px_per_unit=8)