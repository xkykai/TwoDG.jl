using CSV
using DataFrames

function make_circle_mesh(size)
    TEMP_DIR = "$(@__DIR__)/temp_circle_$(size)"
    mkpath(TEMP_DIR)

    command = `python pyscripts/make_circle_mesh.py $(size) $(TEMP_DIR)`
    run(command)

    p = Array(CSV.read("$(TEMP_DIR)/p.csv", DataFrame, header=false))
    t = Array(CSV.read("$(TEMP_DIR)/t.csv", DataFrame, header=false))

    t .+= 1

    # @info "Removing mesh files"
    rm(TEMP_DIR, recursive=true)
    return p, t
end

function make_square_mesh(m, n, parity)
    TEMP_DIR = "$(@__DIR__)/temp_square_$(m)_$(n)_$(parity)"
    mkpath(TEMP_DIR)

    command = `python pyscripts/make_square_mesh.py $(m) $(n) $(parity) $(TEMP_DIR)`
    run(command)

    p = Array(CSV.read("$(TEMP_DIR)/p.csv", DataFrame, header=false))
    t = Array(CSV.read("$(TEMP_DIR)/t.csv", DataFrame, header=false))

    t .+= 1

    rm(TEMP_DIR, recursive=true)
    return p, t
end