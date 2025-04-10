using TwoDG
using TwoDG.Masters: shape1d
using Statistics
using CairoMakie

nodetype = 1

time_initialize = 10
time_final = 20
dt = 6e-3
nstep = 20
ncycl_initialize = Int(ceil(time_initialize / (nstep * dt)))
ncycl_equilibrated = Int(ceil((time_final - time_initialize) / (nstep * dt)))

m = 11
n = 20
c = 1
k = [3, 0]
kmod = sqrt(sum(k.^2))
tparam = [0, 0, 1]

porder = 5
mesh = mkmesh_trefftz(m, n, porder, nodetype, tparam)
master = Master(mesh, 2*porder)
interp_points = 0.05:0.1:0.95
n_interp = length(interp_points)

a = rand(6)
a_interp = shapefunction[:, 1, :]' * a
#%%
fig = Figure()
ax = Axis(fig[1, 1], aspect=1)
scatter!(ax, master.ploc1d[:, 2], a)
scatter!(ax, interp_points, a_interp)
display(fig)
#%%

θs_p = []
ηs_p = []

porders = [4]
for porder in porders
    mesh = mkmesh_trefftz(m, n, porder, nodetype, tparam)
    master = Master(mesh, 2*porder)
    shapefunction = shape1d(porder, master.ploc1d, interp_points)

    indp_rad = findall(i -> sqrt(sum(mesh.p[i, :].^2)) ≈ ℯ, 1:size(mesh.p, 1))
    indf_rad = findall(i -> mesh.f[i, 1] in indp_rad && mesh.f[i, 2] in indp_rad, 1:size(mesh.f, 1))

    ng1d = length(master.gw1d)
    θs = zeros(length(indf_rad) * length(a_interp))
    permls = zeros(Int, length(θs))
    els = zeros(Int, length(θs))
    ηs = zeros(length(θs))

    iθ = 1
    for i in indf_rad
        ipt = mesh.f[i, 1] + mesh.f[i, 2]
        el = mesh.f[i, 3]

        ipl = sum(mesh.t[el, :]) - ipt
        isl = findfirst(x -> x == ipl, mesh.t[el, :])
        iol = mesh.t2f[el, isl] < 0 ? 2 : 1  # Adjusted from 0/1 to 1/2
        perml = master.perm[:, isl, iol]
        dgs = mesh.dgnodes[perml, :, el]

        θ = atan.(dgs[:, 2], dgs[:, 1])

        θs[iθ:iθ + ng1d - 1] .= θ
        permls[iθ:iθ + ng1d - 1] .= perml
        els[iθ:iθ + ng1d - 1] .= el
        iθ += ng1d
    end

    θs[θs .< 0] .+= 2π

    θ_perm = sortperm(θs)
    θs .= θs[θ_perm]
    permls .= permls[θ_perm]
    els .= els[θ_perm]

    bcm = [2, 3]
    bcs = zeros(3, 3)
    app = mkapp_wave()
    app = App(app; bcm, bcs)

    ub(c, k, p, time) = sin.(p[:,1] .* k[1] .+ p[:,2] .* k[2] .- c * sqrt(k[1]^2 + k[2]^2) * time)

    app.pg = true
    app.arg[:c] = c
    app.arg[:k] = k
    app.arg[:f] = ub

    u = initu(mesh, app, [0, 0, 0])
    ue = zeros(size(u))

    @info "Initializing the solution..."
    tm = 0
    for i in 1:ncycl_initialize
        # Update `ue` based on the current time `tm`
        ue[:, 3, :] .= sin.(mesh.dgnodes[:, 1, :] .* k[1] .+ mesh.dgnodes[:, 2, :] .* k[2] .- c * kmod * tm)
        ue[:, 1, :] .= -k[1] * ue[:, 3, :] / kmod
        ue[:, 2, :] .= -k[2] * ue[:, 3, :] / kmod

        # Set initial condition for `u` if it's the first iteration
        if i == 1
            u .= ue
        end

        # # Plot the current solution
        # # fig = scaplot(mesh, u[:, 3, :])
        # if i != 1
        #     fig = scaplot(mesh, u[:, 3, :] .- sin.(mesh.dgnodes[:, 1, :] .* k[1] .+ mesh.dgnodes[:, 2, :] .* k[2] .- c * kmod * tm))
        #     display(fig)
        # end

        # Update the solution using the convection operator
        rk4!(rinvexpl, master, mesh, app, u, tm, dt, nstep)

        # Increment time
        tm += nstep * dt
    end

    scaplot(mesh, u[:, 3, :] .- sin.(mesh.dgnodes[:, 1, :] .* k[1] .+ mesh.dgnodes[:, 2, :] .* k[2] .- c * kmod * tm))

    @info "Computing wave amplitude..."
    for i in 1:ncycl_equilibrated
        # Update `ue` based on the current time `tm`
        ue[:, 3, :] .= sin.(mesh.dgnodes[:, 1, :] .* k[1] .+ mesh.dgnodes[:, 2, :] .* k[2] .- c * kmod * tm)
        ue[:, 1, :] .= -k[1] * ue[:, 3, :] / kmod
        ue[:, 2, :] .= -k[2] * ue[:, 3, :] / kmod

        # Update the solution using the convection operator
        rk4!(rinvexpl, master, mesh, app, u, tm, dt, nstep)

        for i in eachindex(θs)
            el = els[i]
            perm = permls[i]

            η = abs(u[perm, 3, el] - sin(mesh.dgnodes[perm, 1, el] * k[1] + mesh.dgnodes[perm, 2, el] * k[2] - c * kmod * tm))

            ηs[i] = η > ηs[i] ? η : ηs[i]
        end

        # Increment time
        tm += nstep * dt
    end

    scaplot(mesh, u[:, 3, :] .- sin.(mesh.dgnodes[:, 1, :] .* k[1] .+ mesh.dgnodes[:, 2, :] .* k[2] .- c * kmod * tm))

    push!(θs_p, θs)
    push!(ηs_p, ηs)
end
#%%
θ_perm = sortperm(θs_p[1])
lines(θs_p[1][θ_perm] ./ π, abs.(ηs_p[1][θ_perm]), label="p = $(porders[1])")

scatter(θs_p[1], zeros(length(ηs_p[1])))
#%%