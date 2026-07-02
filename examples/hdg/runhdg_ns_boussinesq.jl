#=
Natural convection in a differentially heated square cavity: the 2D
incompressible nonhydrostatic Navier-Stokes equations under the Boussinesq
approximation, solved with the HDG method (Nguyen, Peraire & Cockburn, JCP 2011).

Nondimensionalized with the buoyancy velocity U = sqrt(gβΔT L):

    ∂u/∂t + ∇·(u⊗u) = -∇p + sqrt(Pr/Ra) ∇²u + θ ĵ
    ∇·u = 0
    ∂θ/∂t + ∇·(uθ)  = 1/sqrt(Ra Pr) ∇²θ

on the unit square with no-slip walls, hot left wall (θ = +1/2), cold right
wall (θ = -1/2), and insulated top/bottom walls. The momentum and continuity
equations are advanced with backward Euler and Newton sub-iterations
(hdg_ns_step); the temperature is advanced implicitly with the scalar HDG
transport solver (hdg_cd_step), using the HDG velocity trace as the (exactly
conservative) face convective velocity.

The steady state is validated against the de Vahl Davis benchmark: the average
hot-wall Nusselt number is Nu = 1.118 at Ra = 10³ and Nu = 2.243 at Ra = 10⁴.
=#
using TwoDG
using LinearAlgebra
using Printf
using CairoMakie

#%% Parameters
Ra = 1e4
Pr = 0.71
ν = sqrt(Pr / Ra)              # momentum diffusivity (buoyancy-velocity scaling)
κ = 1 / sqrt(Ra * Pr)          # thermal diffusivity

n = 24                         # n × n × 2 triangles
porder = 3
Δt = 0.25
nsteps = 400
nnewton = 2                    # Newton sub-iterations per time step
steady_tol = 1e-6              # ‖uⁿ⁺¹-uⁿ‖/Δt threshold to stop early

τ = 1.0                        # HDG stabilization (≈ ν/ℓ + |u|, |u| = O(1))

mesh = mkmesh_square(n + 1, n + 1, porder, 0, 1)
master = Master(mesh, 3 * (porder + 1))
npl, nt = size(mesh.dgnodes, 1), size(mesh.t, 1)
nps, nf = porder + 1, size(mesh.f, 1)

# Boundary conditions. Square mesh boundary tags: 1 bottom, 2 right, 3 top, 4 left.
dbc(p) = [0.0, 0.0]                                       # no-slip everywhere
function tbc(p, tag)
    tag == 4 && return (:d, 0.5)                          # hot left wall
    tag == 2 && return (:d, -0.5)                         # cold right wall
    return (:n, 0.0)                                      # insulated top/bottom
end

#%% Hot-wall Nusselt number from the temperature gradient q = ∇θ
function nusselt_hot(mesh, master, q)
    nps = mesh.porder + 1
    sh1d = master.sh1d[:, 1, :]
    sh1dx = master.sh1d[:, 2, :]
    Nu = 0.0
    for i in axes(mesh.f, 1)
        mesh.f[i, 4] == -4 || continue
        it = mesh.f[i, 3]
        lf = findfirst(x -> abs(x) == i, mesh.t2f[it, :])
        pp = master.perm[:, lf, 1]
        xξ = sh1dx' * mesh.dgnodes[pp, 1, it]
        yξ = sh1dx' * mesh.dgnodes[pp, 2, it]
        ds = sqrt.(xξ .^ 2 .+ yξ .^ 2)
        q1g = sh1d' * q[pp, 1, it]
        Nu -= sum(master.gw1d .* ds .* q1g)
    end
    return Nu
end

#%% Time stepping (operator splitting: implicit θ transport, then NS step)
θ = [0.5 - mesh.dgnodes[k, 1, it] for k in 1:npl, it in 1:nt]   # conductive profile
u = zeros(npl, 2, nt)
Λ = zeros(2 * nps * nf)
q = zeros(npl, 2, nt)
dtinv = 1 / Δt

t = 0.0
hist_t, hist_nu, hist_ke = Float64[], Float64[], Float64[]

for step in 1:nsteps
    global θ, u, Λ, q, t

    # temperature advanced with the velocity from the previous time level
    θres = hdg_cd_step(master, mesh, κ, tbc; τ=τ, u, Λ, θold=θ, dtinv)
    θ = θres.θ
    q = θres.q

    # Navier-Stokes step with buoyancy source (0, θ)
    src = zeros(npl, 2, nt)
    src[:, 2, :] .= θ
    uold = copy(u)
    local res
    for inner in 1:nnewton
        res = hdg_ns_step(master, mesh, ν, dbc; τ, source=src, u, Λ, uold, dtinv)
        u, Λ = res.u, res.Λ
    end

    t += Δt
    Δu = norm(u .- uold) / max(norm(u), eps()) / Δt
    Nu = nusselt_hot(mesh, master, q)
    ke = sum(u .^ 2)
    push!(hist_t, t); push!(hist_nu, Nu); push!(hist_ke, ke)

    if step % 10 == 0 || step == 1
        @info @sprintf("step %4d  t = %6.2f  Nu_hot = %.4f  ‖Δu‖/Δt = %.2e", step, t, Nu, Δu)
    end
    if Δu < steady_tol
        @info @sprintf("reached steady state at t = %.2f (step %d)", t, step)
        break
    end
end

Nu = nusselt_hot(mesh, master, q)
Nu_ref = Ra ≈ 1e3 ? 1.118 : Ra ≈ 1e4 ? 2.243 : NaN
@info @sprintf("steady hot-wall Nusselt number: %.4f (benchmark %.3f)", Nu, Nu_ref)

#%% Plots
speed = sqrt.(u[:, 1, :] .^ 2 .+ u[:, 2, :] .^ 2)
figθ = scaplot(mesh, θ, show_mesh=false, title="Temperature θ, Ra = $(Int(Ra)), Pr = $Pr", cmap=Reverse(:RdBu))
save("./output/hdg_ns_boussinesq_temperature.png", figθ, px_per_unit=4)
figu = scaplot(mesh, speed, show_mesh=false, title="Speed |u|, Ra = $(Int(Ra)), Pr = $Pr")
save("./output/hdg_ns_boussinesq_speed.png", figu, px_per_unit=4)
display(figθ)
display(figu)

fig = Figure(size=(900, 400))
ax1 = Axis(fig[1, 1], xlabel="t", ylabel="Nu (hot wall)", title="Nusselt number history")
lines!(ax1, hist_t, hist_nu, linewidth=3)
hlines!(ax1, [Nu_ref], linestyle=:dash, color=:gray)
ax2 = Axis(fig[1, 2], xlabel="t", ylabel="ΣU²", title="Kinetic energy history")
lines!(ax2, hist_t, hist_ke, linewidth=3)
display(fig)
save("./output/hdg_ns_boussinesq_history.png", fig, px_per_unit=4)
