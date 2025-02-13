using CSV
using DataFrames

function make_circle_mesh(size)
    TEMP_DIR = "$(@__DIR__)/temp_$(size)"
    mkpath(TEMP_DIR)

    # @info "Making circle mesh with size $(size) using python"
    command = `python pyscripts/make_circle_mesh.py $(size) $(TEMP_DIR)`
    run(command)

    p = Array(CSV.read("$(TEMP_DIR)/p.csv", DataFrame, header=false))
    t = Array(CSV.read("$(TEMP_DIR)/t.csv", DataFrame, header=false))

    # @info "Removing mesh files"
    rm(TEMP_DIR, recursive=true)
    return p, t
end