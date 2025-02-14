using TwoDG.Util:unique_rows

"""
    simpvol(p::Matrix{T}, t::Matrix{Int}) where T<:Real

Compute signed volumes (areas) of triangular elements in a 2D mesh.

Parameters:
- `p`: Nx2 matrix of vertex coordinates where N is the number of vertices
- `t`: Mx3 matrix of triangle vertex indices where M is the number of triangles

Returns:
- Vector of signed volumes (areas) for each triangle

Performance optimizations:
- Uses views instead of array copies for index operations
- Preallocates output array
- Leverages SIMD operations through Julia's native array operations
"""
function simpvol(p::Matrix{T}, t::Matrix{Int}) where T<:Real
    # Get number of triangles
    num_triangles = size(t, 1)
    
    # Preallocate output array
    volumes = Vector{T}(undef, num_triangles)
    
    # Use views for efficient indexing
    @views for i in 1:num_triangles
        # Extract vertex indices for current triangle
        v0, v1, v2 = t[i, 1], t[i, 2], t[i, 3]
        
        # Calculate edge vectors
        d01_x = p[v1, 1] - p[v0, 1]
        d01_y = p[v1, 2] - p[v0, 2]
        d02_x = p[v2, 1] - p[v0, 1]
        d02_y = p[v2, 2] - p[v0, 2]
        
        # Calculate signed volume (area)
        volumes[i] = (d01_x * d02_y - d01_y * d02_x) / 2
    end
    
    return volumes
end

"""
    fixmesh(p::Matrix{T}, t::Matrix{Int}, ptol::Real=2e-13) where T<:Real

Remove duplicated/unused nodes and fix element orientation in a mesh.

Parameters:
- `p`: Nx2 matrix of vertex coordinates
- `t`: Mx3 matrix of triangle vertex indices
- `ptol`: tolerance for identifying duplicate vertices (default: 2e-13)

Returns:
- Tuple of (cleaned vertex matrix, fixed triangle matrix)
"""
function fixmesh(p::Matrix{T}, t::Matrix{Int}, ptol::Real=2e-13) where T<:Real
    # Find scaling factor for snapping based on mesh size
    snap = maximum(maximum(p, dims=1) - minimum(p, dims=1)) * ptol
    
    # Fix nearly-zero coordinates (handling -0.0 vs 0.0 issue)
    zero_idx = findall(abs.(p) .< snap)
    p[zero_idx] .= zero(T)
    
    # Round coordinates to snap threshold and find unique vertices
    p_rounded = round.(p ./ snap) .* snap
    p_unique, ix, jx = unique_rows(p_rounded; return_index=true, return_inverse=true)
    
    t_unique = similar(t)
    for i in eachindex(t)
        t_unique[i] = jx[t[i]]
    end
    
    # Fix triangle orientation based on signed volume
    vols = simpvol(p, t)
    flip_idx = findall(vols .< 0)
    
    # Flip triangles with negative volume
    if !isempty(flip_idx)
        t[flip_idx, 1:2] = t[flip_idx, 2:-1:1]
    end
    
    return p_unique, t_unique
end


"""
mkt2t(t)
Compute element connectivities from element indices.

t2t, t2n = mkt2t(t)
"""
function mkt2f(t)

end

