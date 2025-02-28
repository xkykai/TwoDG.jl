using TwoDG
using CairoMakie
using DataFrames
using CSV

sizs = [0.4, 0.2, 0.1]
porders = [1, 2, 3, 4]

area1s = zeros(length(sizs), length(porders))
area2s = zeros(length(sizs), length(porders))
perims = zeros(length(sizs), length(porders))

for i in eachindex(sizs), j in eachindex(porders)
    @info "Computing area for size $(sizs[i]) and porder $(porders[j])"
    area1s[i, j], area2s[i, j], perims[i, j] = areacircle(sizs[i], porders[j])
end

#%%
fig = Figure(size=(1200, 400))
ax1 = Axis(fig[1, 1], xlabel="p", title="|Area 1 - π|", yscale=log10)
ax2 = Axis(fig[1, 2], xlabel="p", title="|Area 2 - π|", yscale=log10)
ax3 = Axis(fig[1, 3], xlabel="p", title="|Perimeter - 2π|", yscale=log10)

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
save("./output/area_perimeter_difference.pdf", fig)
#%%
df1 = DataFrame(area1s', :auto)
rename!(df1, Symbol.(sizs))

df2 = DataFrame(area2s', :auto)
rename!(df2, Symbol.(sizs))

df3 = DataFrame(perims', :auto)
rename!(df3, Symbol.(sizs))
#%%
CSV.write("./output/area1.csv", df1)
CSV.write("./output/area2.csv", df2)
CSV.write("./output/perimeter.csv", df3)
#%%