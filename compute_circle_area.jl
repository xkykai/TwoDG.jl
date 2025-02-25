using TwoDG
using CairoMakie

sizs = [0.4, 0.2, 0.1]
porders = [1, 2, 3]

area1s = zeros(length(sizs), length(porders))
area2s = zeros(length(sizs), length(porders))
perims = zeros(length(sizs), length(porders))

for i in eachindex(sizs), j in eachindex(porders)
    @info "Computing area for size $(sizs[i]) and porder $(porders[j])"
    area1s[i, j], area2s[i, j], perims[i, j] = areacircle(sizs[i], porders[j])
end

#%%
fig = Figure(size=(1200, 400))
ax1 = Axis(fig[1, 1], xlabel="p", ylabel="Area difference", title="Area difference vs element order, method 1", yscale=log10)
ax2 = Axis(fig[1, 2], xlabel="p", ylabel="Area difference", title="Area difference vs element order, method 2", yscale=log10)
ax3 = Axis(fig[1, 3], xlabel="p", ylabel="Perimeter difference", title="Perimeter difference vs element order", yscale=log10)

for j in eachindex(sizs)
    lines!(ax1, porders, abs.(area1s[j, :] .- π), label="Size = $(sizs[j])")
end
axislegend(ax1, position=:lb)

for j in eachindex(sizs)
    lines!(ax2, porders, abs.(area2s[j, :] .- π), label="Size = $(sizs[j])")
end
axislegend(ax2, position=:lb)

for j in eachindex(sizs)
    lines!(ax3, porders, abs.(perims[j, :] .- 2π), label="Size = $(sizs[j])")
end
axislegend(ax3, position=:lb)

display(fig)
save("./output/area_perimeter_difference.png", fig, px_per_unit=4)
save("./output/area_perimeter_difference.pdf", fig)
#%%