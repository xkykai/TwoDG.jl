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

interp_points = 0.05:0.1:0.95
n_interp = length(interp_points)

porders = [4, 5]
θs_p = []
ηs_p = []

for porder in porders
    @info "Running for porder = $porder"
    mesh = mkmesh_trefftz(m, n, porder, nodetype, tparam)
    master = Master(mesh, 2*porder)
    shapefunction = shape1d(porder, master.ploc1d, interp_points)

    indp_rad = findall(i -> sqrt(sum(mesh.p[i, :].^2)) ≈ ℯ, 1:size(mesh.p, 1))
    indf_rad = findall(i -> mesh.f[i, 1] in indp_rad && mesh.f[i, 2] in indp_rad, 1:size(mesh.f, 1))

    ng1d = length(master.gw1d)
    θs = zeros(length(indf_rad) * n_interp)
    ηs = similar(θs)
    permls = zeros(Int, length(indf_rad), size(master.ploc1d, 1))
    els = zeros(Int, length(indf_rad))

    iθ = 1
    for (i, face_num) in enumerate(indf_rad)
        ipt = mesh.f[face_num, 1] + mesh.f[face_num, 2]
        el = mesh.f[face_num, 3]

        ipl = sum(mesh.t[el, :]) - ipt
        isl = findfirst(x -> x == ipl, mesh.t[el, :])
        iol = mesh.t2f[el, isl] < 0 ? 2 : 1  # Adjusted from 0/1 to 1/2
        perml = master.perm[:, isl, iol]
        dgs = mesh.dgnodes[perml, :, el]

        dgs_interp = shapefunction[:, 1, :]' * dgs

        θ = atan.(dgs_interp[:, 2], dgs_interp[:, 1])

        θs[iθ:iθ + n_interp - 1] .= θ
        permls[i, :] .= perml
        els[i] = el
        iθ += n_interp
    end

    θs[θs .< 0] .+= 2π

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

        # Update the solution using the convection operator
        rk4!(rinvexpl, master, mesh, app, u, tm, dt, nstep)

        # Increment time
        tm += nstep * dt
    end

    # scaplot(mesh, u[:, 3, :] .- sin.(mesh.dgnodes[:, 1, :] .* k[1] .+ mesh.dgnodes[:, 2, :] .* k[2] .- c * kmod * tm))

    @info "Computing wave amplitude..."
    for i in 1:ncycl_equilibrated
        # Update `ue` based on the current time `tm`
        ue[:, 3, :] .= sin.(mesh.dgnodes[:, 1, :] .* k[1] .+ mesh.dgnodes[:, 2, :] .* k[2] .- c * kmod * tm)
        ue[:, 1, :] .= -k[1] * ue[:, 3, :] / kmod
        ue[:, 2, :] .= -k[2] * ue[:, 3, :] / kmod

        # Update the solution using the convection operator
        rk4!(rinvexpl, master, mesh, app, u, tm, dt, nstep)

        iθ = 1
        for (i, face_num) in enumerate(indf_rad)
            el = els[i]
            perml = permls[i, :]
            us = u[perml, 3, el]
            ηs_interp = abs.(shapefunction[:, 1, :]' * us)

            η = @view ηs[iθ:iθ + n_interp - 1]
            η .= (ηs_interp .> η) .* ηs_interp .+ (ηs_interp .<= η) .* η

            iθ += n_interp
        end

        # Increment time
        tm += nstep * dt
    end

    fig = scaplot(mesh, u[:, 3, :] .- sin.(mesh.dgnodes[:, 1, :] .* k[1] .+ mesh.dgnodes[:, 2, :] .* k[2] .- c * kmod * tm))
    display(fig)

    push!(θs_p, θs)
    push!(ηs_p, ηs)
end
#%%
fig = Figure()
ax = Axis(fig[1, 1], xlabel="θ / π", ylabel="|η|", title="Scattered wave amplitude")
for (i, porder) in enumerate(porders)
    θ_perm = sortperm(θs_p[i])
    lines!(ax, θs_p[i][θ_perm] ./ π, abs.(ηs_p[i][θ_perm]), label="p = $(porder)")
end
axislegend(ax, position=:ct)
display(fig)
save("./output/scattered_wave_amplitude.png", fig, px_per_unit=8)
#%%