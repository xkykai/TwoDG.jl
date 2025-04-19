using TwoDG
using CairoMakie
using Statistics
using LinearAlgebra

nodetype = 1

kappa = 1
c11 = 10
c11int = 0

app = mkapp_convection_diffusion()
app.pg = true

function vf(p)
    return zero(p)
end

function src(ug, args...)
    return zero(ug) .+ 1
end

bcm = [1]
bcs = zeros(1, 1)
app = App(app; bcm, bcs, src)
app.arg[:vf] = vf
app.arg[:kappa] = kappa
app.arg[:c11] = c11
app.arg[:c11int] = c11int


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
    u = initu(mesh, app, [0])

    time = 0
    for i in 1:ncycl
        @info "time = $(time)"
        rk4!(rldgexpl, master, mesh, app, u, time, dt, nstep)
        time += nstep * dt
    end

    fig = scaplot(mesh, u[:, 1, :], show_mesh=true)
    display(fig)
    l2_errors[i][j] = l2_error(mesh, u, poisson_solution)
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