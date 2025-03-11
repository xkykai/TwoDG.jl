using CSV
using DataFrames

# Use Python's distmesh to generate a mesh. 
# Required packages in python environment:
# distmesh, numpy
# distmesh has also been provided locally
# This is a hack: the meshes generated by the distmesh are saved to a temporary directory and then read back in Julia
function make_circle_mesh(size)
    TEMP_DIR = "$(@__DIR__)/temp_circle_$(size)"
    mkpath(TEMP_DIR)

    command = `python pyscripts/make_circle_mesh.py $(size) $(TEMP_DIR)`
    run(command)

    p = Array(CSV.read("$(TEMP_DIR)/p.csv", DataFrame, header=false))
    t = Array(CSV.read("$(TEMP_DIR)/t.csv", DataFrame, header=false))

    t .+= 1

    rm(TEMP_DIR, recursive=true)
    return p, t
end

"""
make_square_mesh 2-d regular triangle mesh generator for the unit square
p, t = squaremesh(m, n, parity)

  p:         node positions (np,2)
  t:         triangle indices (nt,3)
  parity:    flag determining the triangular pattern
              flag = 0 (diagonals sw - ne) (default)
              flag = 1 (diagonals nw - se)
"""
function make_square_mesh(m::Int=10, n::Int=10, parity::Int=0)
    # Generate mesh for unit square
    x = range(0.0, 1.0, length=m)
    y = range(0.0, 1.0, length=n)
    
    # Create meshgrid equivalent in Julia
    X = [x[i] for j in 1:n, i in 1:m]
    Y = [y[j] for j in 1:n, i in 1:m]
    
    # Create node positions
    p = hcat(X[:], Y[:])
    
    # Pre-allocate triangle indices
    nt = 2*(m-1)*(n-1)
    t = zeros(Int, nt, 3)
    
    # Generate triangle indices
    idx = 1
    for j in 1:(n-1), i in 1:(m-1)
        i0 = i + (j-1)*m
        i1 = i0 + 1
        i2 = i0 + m
        i3 = i2 + 1
        
        if parity == 0
            t[idx,:] = [i0, i3, i2]
            t[idx+1,:] = [i0, i1, i3]
        else
            t[idx,:] = [i0, i1, i2]
            t[idx+1,:] = [i1, i3, i2]
        end
        
        idx += 2
    end
    
    return p, t
end