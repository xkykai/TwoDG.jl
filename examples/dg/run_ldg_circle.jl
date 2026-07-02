using TwoDG
using CairoMakie
using Statistics
using LinearAlgebra
using StaticArrays

nodetype = 1

kappa = 1
c11 = 10
c11int = 0

# unit source term (pointwise convention)
src(u, x, param, time) = SVector(1.0)

bcm = [1]
bcs = zeros(1, 1)
app = mkapp_convection_diffusion_pt(SVector(0.0, 0.0);
                                    kappa, c11, c11int, bcm, bcs, src)


time_total = 2
nstep = 32
dt = time_total / (nstep * 250)
ncycl = Int(ceil(time_total / dt / nstep))

function poisson_solution(x, y)
    r² = x^2 + y^2
    return 0.25 * (1 - r²)
end
#%%
porders = [1, 2]
sizes = [0.4, 0.25]

l2_errors = [zeros(length(sizes)) for _ in porders]
for (i, porder) in enumerate(porders), (j, size) in enumerate(sizes)
    @info "Computing MSE for p = $(porder) and size = $(size)"
    mesh = mkmesh_circle(size, porder, 1)
    master = Master(mesh, 4*porder)
    ctx = DGContext(master, mesh)
    u = initu(mesh, app, [0])

    time = 0
    for i in 1:ncycl
        @info "time = $(time)"
        rk4_ka!(rldgexpl!, ctx, app, u, time, dt, nstep)
        time += nstep * dt
    end

    fig = scaplot(mesh, u[:, 1, :], show_mesh=true)
    display(fig)
    l2_errors[i][j] = l2error(mesh, u, poisson_solution)
end
#%%
fig = Figure()
ax = Axis(fig[1, 1], xlabel="1/h", ylabel="L2 norm", title="Error convergence of LDG, circular domain", yscale=log10, xscale=log2)
for i in eachindex(porders)
    scatterlines!(ax, 1 ./ sizes, l2_errors[i], label="p = $(porders[i])")
end
axislegend(ax, position=:lb)
display(fig)
# save("./output/ldg_circle_convergence.png", fig, px_per_unit=8)
#%%