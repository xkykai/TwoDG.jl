using TwoDG.Utils:unique_rows

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
function mkt2f(t::Matrix{Int})
    # Number of triangles
    nt = size(t, 1)
    # Matrix to store all edges (3 per triangle)
    all_faces = zeros(Int, nt * 3, 2)
    for i in axes(t, 1)
        # Extract and sort vertices for each edge of the triangle
        all_faces[i, :] .= sort(t[i, [1, 2]])
        all_faces[i + nt, :] .= sort(t[i, [2, 3]])
        all_faces[i + 2nt, :] .= sort(t[i, [3, 1]])
    end
    
    # Create mapping from ordered edge vertices to triangle indices
    # This stores which triangle contains each edge in its original orientation
    face_lt_map = Dict{Tuple{Int, Int}, Int}()
    sizehint!(face_lt_map, 3*nt)
    for i in axes(t, 1)
        face_lt_map[Tuple(t[i, 1:2])] = i
        face_lt_map[Tuple(t[i, 2:3])] = i
        face_lt_map[Tuple(t[i, [3, 1]])] = i
    end
    
    # Count occurrences of each edge to identify interior vs. boundary edges
    row_counts = Dict{Tuple{Int, Int}, Int}()
    sizehint!(row_counts, 3*nt)
    for row in eachrow(all_faces)
        row_counts[Tuple(row)] = get(row_counts, Tuple(row), 0) + 1
    end
    
    # Identify boundary edges (those appearing only once in the mesh)
    boundary_faces = [key for (key, val) in row_counts if val == 1]
    nb = length(boundary_faces)
    # Calculate total number of unique edges
    nf = (3*nt + nb) ÷ 2
    
    # Initialize face matrix: [vertex1, vertex2, triangle1, triangle2]
    # For boundary edges, triangle2 will be 0
    f = zeros(Int, nf, 4)
    f_ii = 1  # Index for interior edges
    f_ib = nf - nb + 1  # Starting index for boundary edges
    f_bn = -1  # Used for tracking boundary connectivity
    
    # Data structures for tracking boundary topology (connected components)
    face_topology = Vector{Set{Int}}()
    sizehint!(face_topology, 1)
    cclockwise_boundary_faces = Dict{Int, Tuple{Int, Int}}()
    sizehint!(cclockwise_boundary_faces, nb)
    new_set = true
    
    # Populate the face matrix
    for key in keys(row_counts)
        if row_counts[key] != 1
            # Handle interior edges (shared by two triangles)
            f[f_ii, 1:2] .= key
            f[f_ii, 3] = face_lt_map[key]
            f[f_ii, 4] = face_lt_map[reverse(key)]
            f_ii += 1
        else
            # Handle boundary edges (part of only one triangle)
            # Check if edge is in its original orientation in the triangle
            cclock = haskey(face_lt_map, key)
            
            # Try to connect this boundary edge to existing boundary components
            # This helps identify holes or separate boundary loops
            for (i, set) in enumerate(face_topology)
                if key[1] in set || key[2] in set
                    push!(set, key...)
                    new_set = false
                    f_bn = -i
                    break
                end
            end
            
            # If not connected to existing components, create a new one
            if new_set
                push!(face_topology, Set(key))
                f_bn = -length(face_topology)
            end
            
            # Store boundary edge information with proper orientation
            if cclock
                cclockwise_boundary_faces[key[1]] = (key[2], f_ib)
                f[f_ib, 1:3] = [key..., face_lt_map[key]]
            else
                cclockwise_boundary_faces[key[2]] = (key[1], f_ib)
                f[f_ib, 1:3] = [reverse(key)..., face_lt_map[reverse(key)]]
            end
            new_set = true
            f_ib += 1
        end
    end
    
    # Create lookup table from vertex sets to face indices
    f_lookup = Dict(Set(f[i, 1:2]) => i for i in axes(f, 1))
    
    # Create triangle-to-face mapping with orientation information
    t2f = zeros(Int, nt, 3)
    for i in axes(t, 1), j in axes(t, 2)
        # Determine which edge of the triangle we're processing based on the current vertex
        if j == 1
            look_index = [2, 3]  # Edge opposite to first vertex
        elseif j == 2
            look_index = [3, 1]  # Edge opposite to second vertex
        else
            look_index = [1, 2]  # Edge opposite to third vertex
        end
        
        # Find the corresponding face in our face matrix
        face = t[i, look_index]
        triangle_loc = f_lookup[Set(t[i, look_index])]
        
        # Store face index with sign indicating orientation:
        # Positive if edge orientation matches triangle orientation
        # Negative if edge orientation is reversed relative to triangle
        if face[1] == f[triangle_loc, 1]
            t2f[i, j] = triangle_loc
        else
            t2f[i, j] = -triangle_loc
        end
    end
    
    return f, t2f
end

""" 
p:         Node positions (:,2)
f:         Face Array (:,4)
bndexpr:   Cell Array of boundary expressions. The 
           number of elements in BNDEXPR determines 
           the number of different boundaries

Example: (Setting boundary types for a unit square mesh - 4 types)
bndexpr = [lambda p: np.all(p[:,0]<1e-3, lambda p: np.all(p[:,0]>1-1e-3),
          lambda p: np.all(p[:,1]<1e-3, lambda p: np.all(p[:,1]>1-1e-3)]
f = setbndnbrs(p,f,bndexpr);

Example: (Setting boundary types for the unit circle - 1 type)
bndexpr = [lambda p: np.all(np.sqrt((p**2).sum(1))>1.0-1e-3)] 
f = setbndnbrs(p,f,bndexpr);
"""
function setbndnbrs(p, f, bndexpr)
    # Find the first index where column 4 of f contains 0
    # This marks the beginning of boundary elements in the mesh
    i_bnd = findfirst(x -> x == 0, f[:, 4])
    
    # Calculate midpoints of boundary edges by averaging their endpoint coordinates
    # f[i_bnd:end, 1] and f[i_bnd:end, 2] contain indices of endpoints in point array p
    midpoint = (p[f[i_bnd:end, 1], :] .+ p[f[i_bnd:end, 2], :]) ./ 2
    
    if length(bndexpr) == 1
        # If only one boundary expression is provided, 
        # mark all boundary elements with -1 (single boundary type)
        f[i_bnd:end, 4] .= -1
    else
        # For multiple boundary expressions, classify each boundary element
        for j in 1:length(bndexpr)
            # Create boolean mask: false for non-boundary elements (before i_bnd),
            # then apply boundary condition j to midpoints to determine which elements belong to boundary j
            is_boundary_j = vcat(falses(i_bnd - 1), bndexpr[j](midpoint))
            
            # Mark elements satisfying boundary condition j with value -j
            # Negative values in column 4 indicate boundary types
            f[is_boundary_j, 4] .= -j
        end
    end
    
    # Return the mesh face array with updated boundary classifications
    return f
end