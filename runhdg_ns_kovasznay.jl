#=
Verification of the HDG solver for the steady 2D incompressible Navier-Stokes
equations (Nguyen, Peraire & Cockburn, JCP 2011) with the Kovasznay flow:

    u1 = 1 - exp(λx) cos(2πy)
    u2 = λ/(2π) exp(λx) sin(2πy)
    p  = -exp(2λx)/2 + C,    λ = Re/2 - sqrt(Re²/4 + 4π²)

an exact solution of the homogeneous incompressible Navier-Stokes equations.
Dirichlet boundary conditions are taken from the exact solution on the domain
(0,2) × (-0.5,1.5) at Re = 20 (ν = 0.05), and we verify the optimal k+1
convergence orders of velocity, pressure, and velocity gradient, as well as the
k+2 superconvergence of the exactly divergence-free, H(div)-conforming
postprocessed velocity u*.
=#
using TwoDG
using LinearAlgebra
using CairoMakie

#%% Exact solution
Re = 20.0
ν = 1 / Re
λk = Re / 2 - sqrt(Re^2 / 4 + 4π^2)

u1e(x, y) = 1 - exp(λk * x) * cos(2π * y)
u2e(x, y) = λk / (2π) * exp(λk * x) * sin(2π * y)
pmean = -(exp(4λk) - 1) / (8λk)              # mean of -exp(2λx)/2 over the domain
pe(x, y) = -exp(2λk * x) / 2 - pmean
L11e(x, y) = -λk * exp(λk * x) * cos(2π * y)
L12e(x, y) = 2π * exp(λk * x) * sin(2π * y)
L21e(x, y) = λk^2 / (2π) * exp(λk * x) * sin(2π * y)
L22e(x, y) = λk * exp(λk * x) * cos(2π * y)

dbc(p) = [u1e(p[1], p[2]), u2e(p[1], p[2])]

# Map the unit square mesh to (0,2) x (-0.5,1.5)
function transform!(mesh)
    for arr in (mesh.p, mesh.pcg)
        arr[:, 1] .= 2 .* arr[:, 1]
        arr[:, 2] .= 2 .* arr[:, 2] .- 0.5
    end
    mesh.dgnodes[:, 1, :] .= 2 .* mesh.dgnodes[:, 1, :]
    mesh.dgnodes[:, 2, :] .= 2 .* mesh.dgnodes[:, 2, :] .- 0.5
    return mesh
end

#%% Convergence study
ns = [4, 8, 16, 32]
porders = [1, 2, 3]

err_u = [zeros(length(ns)) for _ in porders]
err_p = [zeros(length(ns)) for _ in porders]
err_L = [zeros(length(ns)) for _ in porders]
err_us = [zeros(length(ns)) for _ in porders]
finest = nothing

for (k, porder) in enumerate(porders), (j, n) in enumerate(ns)
    @info "Kovasznay: solving with p = $porder, $n × $n × 2 elements"
    ngauss = 3 * (porder + 1)
    mesh = transform!(mkmesh_square(n + 1, n + 1, porder, 0, 1))
    master = Master(mesh, ngauss)
    mesh1 = transform!(mkmesh_square(n + 1, n + 1, porder + 1, 0, 1))
    master1 = Master(mesh1, ngauss)

    result = hdg_ns_solve(master, mesh, ν, dbc; τ=1.0, maxiter=12, tol=1e-10, verbose=false)
    ustar = hdg_ns_postprocess(master, mesh, master1, mesh1, result)

    # l2_error returns the squared L2 error
    err_u[k][j] = sqrt(l2_error(mesh, result.u[:, 1, :], u1e) +
                       l2_error(mesh, result.u[:, 2, :], u2e))
    err_p[k][j] = sqrt(l2_error(mesh, result.p, pe))
    err_L[k][j] = sqrt(l2_error(mesh, result.gradu[:, 1, :], L11e) +
                       l2_error(mesh, result.gradu[:, 2, :], L12e) +
                       l2_error(mesh, result.gradu[:, 3, :], L21e) +
                       l2_error(mesh, result.gradu[:, 4, :], L22e))
    err_us[k][j] = sqrt(l2_error(mesh1, ustar[:, 1, :], u1e) +
                        l2_error(mesh1, ustar[:, 2, :], u2e))
    @info "  ‖u-uh‖ = $(err_u[k][j]), ‖p-ph‖ = $(err_p[k][j]), ‖L-Lh‖ = $(err_L[k][j]), ‖u-u*‖ = $(err_us[k][j])"

    if porder == porders[end] && n == ns[end-1]
        global finest = (mesh=mesh, result=result)
    end
end

#%% Report convergence orders
for (k, porder) in enumerate(porders)
    @info "p = $porder convergence orders (u, p, L, u*):"
    for j in 2:length(ns)
        ou = log2(err_u[k][j-1] / err_u[k][j])
        op = log2(err_p[k][j-1] / err_p[k][j])
        oL = log2(err_L[k][j-1] / err_L[k][j])
        ous = log2(err_us[k][j-1] / err_us[k][j])
        @info "  n = $(ns[j]):  $(round(ou, digits=2)), $(round(op, digits=2)), $(round(oL, digits=2)), $(round(ous, digits=2))"
    end
end

#%% Convergence plot
fig = Figure(size=(1900, 520))
titles = ["velocity (order k+1)", "pressure (order k+1)", "velocity gradient (order k+1)",
          "postprocessed velocity u* (order k+2)"]
errors = [err_u, err_p, err_L, err_us]
rate(porder, i) = i == 4 ? porder + 2 : porder + 1
colors = Makie.wong_colors()
for (i, (ttl, err)) in enumerate(zip(titles, errors))
    ax = Axis(fig[1, i], xlabel="1/h", ylabel="L² error", title=ttl,
              xscale=log2, yscale=log10)
    for (k, porder) in enumerate(porders)
        scatterlines!(ax, ns ./ 2, err[k], label="k = $porder", linewidth=3, color=colors[k])
        ref = err[k][end] .* (ns[end] ./ ns) .^ rate(porder, i)
        lines!(ax, ns ./ 2, ref, linestyle=:dash, color=colors[k], alpha=0.5)
    end
    i == 1 && axislegend(ax, position=:lb)
end
Label(fig[0, 1:4], "HDG incompressible Navier-Stokes: Kovasznay flow, Re = 20 (dashed: reference order)",
      tellwidth=false, fontsize=20)
display(fig)
save("./output/hdg_ns_kovasznay_convergence.png", fig, px_per_unit=4)

#%% Solution plots on the finest p=3 mesh
mesh, result = finest.mesh, finest.result
speed = sqrt.(result.u[:, 1, :] .^ 2 .+ result.u[:, 2, :] .^ 2)
fig1 = scaplot(mesh, speed, show_mesh=true, title="Kovasznay flow |u|, HDG k=3")
save("./output/hdg_ns_kovasznay_speed.png", fig1, px_per_unit=4)
fig2 = scaplot(mesh, result.p, show_mesh=true, title="Kovasznay flow pressure, HDG k=3")
save("./output/hdg_ns_kovasznay_pressure.png", fig2, px_per_unit=4)
display(fig1)
display(fig2)
