using TwoDG.Utils:unique_rows

"""
    simpvol(p::Matrix{T}, t::Matrix{Int}) where T<:Real

Compute signed volumes of the simplices of a mesh: areas of triangles
(`t (nt, 3)`) or volumes of tetrahedra (`t (nt, 4)`), positive for
counterclockwise / right-handed vertex ordering.

Parameters:
- `p`: N×Dim matrix of vertex coordinates
- `t`: M×(Dim+1) matrix of simplex vertex indices

Returns:
- Vector of signed volumes for each simplex
"""
function simpvol(p::Matrix{T}, t::Matrix{Int}) where T<:Real
    nt = size(t, 1)
    volumes = Vector{T}(undef, nt)

    if size(t, 2) == 3
        @views for i in 1:nt
            v0, v1, v2 = t[i, 1], t[i, 2], t[i, 3]
            d01_x = p[v1, 1] - p[v0, 1]
            d01_y = p[v1, 2] - p[v0, 2]
            d02_x = p[v2, 1] - p[v0, 1]
            d02_y = p[v2, 2] - p[v0, 2]
            volumes[i] = (d01_x * d02_y - d01_y * d02_x) / 2
        end
    else
        @views for i in 1:nt
            v0 = p[t[i, 1], :]
            d1 = p[t[i, 2], :] - v0
            d2 = p[t[i, 3], :] - v0
            d3 = p[t[i, 4], :] - v0
            volumes[i] = (d1[1] * (d2[2] * d3[3] - d2[3] * d3[2]) -
                          d1[2] * (d2[1] * d3[3] - d2[3] * d3[1]) +
                          d1[3] * (d2[1] * d3[2] - d2[2] * d3[1])) / 6
        end
    end

    return volumes
end

"""
    fixmesh(p::Matrix{T}, t::Matrix{Int}, ptol::Real=2e-13) where T<:Real

Remove duplicated nodes and fix element orientation in a simplicial mesh
(triangles or tetrahedra).

Parameters:
- `p`: N×Dim matrix of vertex coordinates
- `t`: M×(Dim+1) matrix of simplex vertex indices
- `ptol`: tolerance for identifying duplicate vertices (default: 2e-13)

Returns:
- Tuple of (cleaned vertex matrix, fixed simplex matrix with positive
  orientation)
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

    # Restore positive orientation: swapping the last two vertices flips the
    # sign of the simplex volume in any dimension
    nv = size(t, 2)
    vols = simpvol(p_unique, t_unique)
    for i in findall(<(0), vols)
        t_unique[i, nv - 1], t_unique[i, nv] = t_unique[i, nv], t_unique[i, nv - 1]
    end

    return p_unique, t_unique
end

"""
    mkt2f(t) -> (f, t2f, t2o)

Face connectivity of a simplicial mesh: triangles `t (nt, 3)` or tetrahedra
`t (nt, 4)` (routed by the column count):

- `f (nf, Dim+2)`: face rows `[v..., left element, right element]` — `Dim`
  vertices, then the two adjacent elements, the right entry `0` for boundary
  faces (later replaced by `-tag`, see [`setbndnbrs`](@ref)). The stored
  vertex order is the **left** element's outward traversal (counterclockwise
  edge in 2D; right-hand-rule outward triangle in 3D).
- `t2f (nt, Dim+1)`: face index of each local face (unsigned; local face `j`
  is opposite local vertex `j`).
- `t2o (nt, Dim+1)`: orientation code of each local face — the index into
  `master.perm[:, s, o]` mapping the face's canonical (stored) node ordering
  to the element's own traversal: `1` when they match (always the case for
  the left element), `2` when reversed in 2D; `1:6` over the triangle
  symmetry group in 3D. This replaces the former sign of `t2f` (an explicit
  small integer is the representation that scales to 3D).
"""
mkt2f(t::Matrix{Int}) = size(t, 2) == 3 ? _mkt2f_tri(t) : _mkt2f_tet(t)

# tetrahedra: faces are sorted vertex triples; interior faces first, then
# boundary faces, both in element-scan order (deterministic)
function _mkt2f_tet(t::Matrix{Int})
    nt = size(t, 1)
    nv = 4

    # gather the (element, local face) pairs sharing each sorted vertex triple
    occ = Dict{NTuple{3, Int}, Vector{NTuple{2, Int}}}()
    sizehint!(occ, 2 * nt)
    for e in 1:nt, j in 1:nv
        fv = face_vertices(Val(3), j)
        key = _sort3(t[e, fv[1]], t[e, fv[2]], t[e, fv[3]])
        push!(get!(() -> NTuple{2, Int}[], occ, key), (e, j))
    end

    ni = count(v -> length(v) == 2, values(occ))
    nf = length(occ)

    f = zeros(Int, nf, 5)
    t2f = zeros(Int, nt, nv)
    t2o = zeros(Int, nt, nv)

    # two passes in element-scan order: interior faces first, then boundary
    assigned = Dict{NTuple{3, Int}, Int}()
    sizehint!(assigned, nf)
    fi, fb = 0, ni
    for e in 1:nt, j in 1:nv
        fv = face_vertices(Val(3), j)
        mine = (t[e, fv[1]], t[e, fv[2]], t[e, fv[3]])
        key = _sort3(mine...)
        idx = get(assigned, key, 0)
        if idx == 0
            interior = length(occ[key]) == 2
            idx = interior ? (fi += 1) : (fb += 1)
            assigned[key] = idx
            # first encounter: this element is the left element; store its
            # outward traversal as the face's canonical direction
            f[idx, 1] = mine[1]
            f[idx, 2] = mine[2]
            f[idx, 3] = mine[3]
            f[idx, 4] = e
            t2o[e, j] = 1
        else
            f[idx, 5] = e
            t2o[e, j] = face_orientation((f[idx, 1], f[idx, 2], f[idx, 3]), mine, Val(3))
        end
        t2f[e, j] = idx
    end

    return f, t2f, t2o
end

@inline function _sort3(a::Int, b::Int, c::Int)
    a > b && ((a, b) = (b, a))
    b > c && ((b, c) = (c, b))
    a > b && ((a, b) = (b, a))
    return (a, b, c)
end

function _mkt2f_tri(t::Matrix{Int})
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
    
    # Create triangle-to-face mapping with explicit orientation codes
    t2f = zeros(Int, nt, 3)
    t2o = zeros(Int, nt, 3)
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

        t2f[i, j] = triangle_loc
        # orientation 1: the element's counterclockwise edge traversal matches
        # the face's stored direction; orientation 2: it is reversed
        t2o[i, j] = face[1] == f[triangle_loc, 1] ? 1 : 2
    end

    return f, t2f, t2o
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
    # the right-element column is the last one; the columns before the two
    # element entries are the face's vertices (2 in 2D, 3 in 3D)
    bcol = size(f, 2)
    nvf = bcol - 2

    # Find the first boundary face (right element still 0)
    i_bnd = findfirst(x -> x == 0, f[:, bcol])

    # Face centroids: average of the vertex coordinates (for 2D this is the
    # edge midpoint, exactly as before)
    midpoint = p[f[i_bnd:end, 1], :]
    for v in 2:nvf
        midpoint = midpoint .+ p[f[i_bnd:end, v], :]
    end
    midpoint = midpoint ./ nvf

    if length(bndexpr) == 1
        # If only one boundary expression is provided,
        # mark all boundary elements with -1 (single boundary type)
        f[i_bnd:end, bcol] .= -1
    else
        # For multiple boundary expressions, classify each boundary face
        for j in 1:length(bndexpr)
            # Create boolean mask: false for interior faces (before i_bnd),
            # then apply classifier j to the centroids
            is_boundary_j = vcat(falses(i_bnd - 1), bndexpr[j](midpoint))

            # Mark faces satisfying boundary condition j with value -j
            f[is_boundary_j, bcol] .= -j
        end
    end

    # Return the mesh face array with updated boundary classifications
    return f
end