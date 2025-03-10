function cgmesh(mesh, ptol=2e-13)
    # Reshape and rearrange mesh.dgnodes
    ph = reshape(permutedims(mesh.dgnodes, (3, 1, 2)), :, 2)
    
    # Create array of indices
    th = reshape(1:size(ph, 1), (size(mesh.dgnodes, 3), :))

    # Find unique rows in rounded ph
    rounded_ph = round.(ph, digits=6)
    
    # Process unique points efficiently with dictionary
    point_tuples = [Tuple(row) for row in eachrow(rounded_ph)]
    unique_dict = Dict{NTuple{2,Float64}, Int}()
    unique_indices = Int[]
    inverse_map = zeros(Int, length(point_tuples))
    
    for (i, point) in enumerate(point_tuples)
        if !haskey(unique_dict, point)
            push!(unique_indices, i)
            unique_dict[point] = length(unique_indices)
        end
        inverse_map[i] = unique_dict[point]
    end
    
    # Extract unique points and update indices
    ph = ph[unique_indices, :]
    th = inverse_map[th]
    
    # Process unique indices
    unique_th = sort(unique(th))
    th_dict = Dict(val => i for (i, val) in enumerate(unique_th))
    th_inverse = [th_dict[val] for val in vec(th)]
    
    # Reshape and finalize
    th = reshape(th_inverse, :, size(mesh.dgnodes, 1))
    ph = ph[unique_th, :]
    
    return Mesh(mesh; mesh.dgnodes, pcg=ph, tcg=th)
end