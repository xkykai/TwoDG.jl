using TwoDG
using CairoMakie

m = 21
n = 11
parity = 0
kappa = 0.01

time_total = 2
nstep = 8
dt = time_total / (nstep * 250)
ncycl = Int(ceil(time_total / dt / nstep))
nper = 5
Re = 100
g = 0.5*Re - sqrt(0.25*Re^2 + 4.0*π^2)

bcm = [2, 2, 2, 1]
bcs = zeros(2, 1)

# Define the Kovasznay flow functions
function xf(p)
    return 1.0 .- exp.(g .* p[:,1]) .* cos.(2.0 .* π .* p[:,2])
end

function yf(p)
    return 0.5 .* g .* exp.(g .* p[:,1]) .* sin.(2.0 .* π .* p[:,2]) ./ π
end

function vf(p)
    return hcat(xf(p), yf(p))
end

function gaus(x, y, s)
    return exp.(-((x .- 1.0).^2 .+ (y .- s).^2) ./ 0.25)
end

function init(x, y)
    return gaus(x, y, 0.0) .+ gaus(x, y, 0.5) .+ gaus(x, y, -0.5)
end

c11 = 10
c11int = 0

porders = [3, 4]

u_locs_p = [zeros(ncycl * nper) for _ in porders]
u_t6_p = []

for (ip, porder) in enumerate(porders)
    @info "order = $(porder)"
    mesh = mkmesh_square(m, n, porder, parity, 1)
    mesh.p[:, 1] = 10.0 .* mesh.p[:, 1]
    mesh.p[:, 2] = 2.5 .* mesh.p[:, 2] .- 1.25
    mesh.dgnodes[:, 1, :] .= 10.0 .* mesh.dgnodes[:, 1, :]
    mesh.dgnodes[:, 2, :] .= 2.5 .* mesh.dgnodes[:, 2, :] .- 1.25

    master = Master(mesh, 3*porder)

    app = mkapp_convection_diffusion()
    app.pg = true

    app = App(app; bcm, bcs)
    app.arg[:vf] = vf
    app.arg[:kappa] = kappa
    app.arg[:c11] = c11
    app.arg[:c11int] = c11int

    elloc = findfirst(i -> sum((mesh.dgnodes[:, 1, i] .≈ 4) .& (mesh.dgnodes[:, 2, i] .≈ 0)) > 0, axes(mesh.dgnodes, 3))
    iploc = findfirst(i -> mesh.dgnodes[i, 1, elloc] ≈ 4 && mesh.dgnodes[i, 2, elloc] ≈ 0, axes(mesh.dgnodes, 1))

    time = 0
    u = initu(mesh, app, [0])

    i_uloc = 1
    for iper in 1:nper
        @info "time = $(time)"
        u .+= initu(mesh, app, [init])

        if iper == 3
            push!(u_t6_p, u)
        end

        fig = scaplot(mesh, u[:, 1, :], show_mesh=true, figure_size=(800, 300), title="t = $(round(time, sigdigits=3)), interior c₁₁ = $(c11int), p = $(porders[ip])")
        display(fig)
        save("./output/conv_diff_t$(round(time, sigdigits=3))_p$porder.png", fig, px_per_unit=8)

        for i in 1:ncycl
            rk4!(rldgexpl, master, mesh, app, u, time, dt, nstep)
            time += nstep * dt

            u_locs_p[ip][i_uloc] = u[iploc, 1, elloc]
            i_uloc += 1
        end
    end
end

#%%
times = range(start=0, stop=10, length=length(u_locs_p[1]))
fig = Figure()
ax = Axis(fig[1, 1], xlabel="t", ylabel="u", title="Concentration at (4, 0), interior c₁₁ = $(c11int)")
for i in eachindex(porders)
    lines!(ax, times, u_locs_p[i], label="p = $(porders[i])")
end
axislegend(ax, position=:lt)
display(fig)
save("./output/conv_diff_conc.png", fig, px_per_unit=8)
#%%
