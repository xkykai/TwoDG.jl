using TwoDG
using CairoMakie
using Statistics

V∞ = 1
α = 10
m = 20
n = 20
porder = 4
node_spacing_type = 1
tparam = [0.1, 0.05, 1.98]
np_foil = 120

mesh, master, xs_foil, ys_foil, chord, ψ, vx, vy, Γ, CP, CF, Clift, Cdrag, CM, vx_analytical, vy_analytical, Γ_analytical, CP_analytical, CL, CL_analytical = trefftz(V∞, α, m, n, porder, node_spacing_type, tparam)

fig = scaplot(mesh, CP, show_mesh=true, title="Pressure Coefficient, α = $(α)°", limits=quantile(CP, [0.01, 0.99]), figure_size=(800, 700))
save("./output/cp_trefftz_$(α).png", fig, px_per_unit=8)

αs = -3:0.5:10
Clifts = zeros(length(αs))
Cdrags = zeros(length(αs))
CLs = zeros(length(αs))
CL_analyticals = zeros(length(αs))
CMs = zeros(length(αs))

for (i, α) in enumerate(αs)
    @info "Computing α = $(α)"
    mesh, master, xs_foil, ys_foil, chord, ψ, vx, vy, Γ, CP, CF, Clift, Cdrag, CM, vx_analytical, vy_analytical, Γ_analytical, CP_analytical, CL, CL_analytical = trefftz(V∞, α, m, n, porder, node_spacing_type, tparam)
    Clifts[i] = Clift
    Cdrags[i] = Cdrag
    CLs[i] = CL
    CL_analyticals[i] = CL_analytical
    CMs[i] = CM
end

#%%
fig = Figure(size=(1200, 400))
axCL = Axis(fig[1, 1], xlabel="α (°)", ylabel="Lift coefficient", title="Lift Coefficient vs Angle of Attack")
axCD = Axis(fig[1, 2], xlabel="α (°)", ylabel="Drag coefficient", title="Drag Coefficient vs Angle of Attack")
axCM = Axis(fig[1, 3], xlabel="α (°)", ylabel="Moment coefficient", title="Moment Coefficient about 1/4 Chord vs \nAngle of Attack")

lines!(axCL, αs, Clifts, label="FE integration")
lines!(axCL, αs, CL_analyticals, label="Theory (analytical Γ)")
lines!(axCL, αs, CLs, label="Theory (FE integrated Γ)")
lines!(axCD, αs, Cdrags, label="FE computation")
lines!(axCM, αs, CMs, label="FE computation")
axislegend(axCL, position=:rb)
axislegend(axCD, position=:lb)
axislegend(axCM, position=:lb)
display(fig)
save("./output/cl_cd_cm_trefftz.pdf", fig)
#%%