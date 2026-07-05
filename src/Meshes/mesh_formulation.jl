using ForwardDiff
using LinearAlgebra
using TwoDG.Utils: newton_raphson

"""
    Mesh

Solver-ready triangular mesh: vertex geometry, DG connectivity, and
high-order (possibly curved) nodes. Built by the `mkmesh_*` generators or by
[`discretize`](@ref)ing a [`MeshGeometry`](@ref); every mesh those return has
`p`–`f2f` fully populated (`pcg`/`tcg` are filled by [`cgmesh`](@ref), which
`discretize` runs for you). The keyword constructor's `nothing` defaults are
construction-internal staging only — solver code must never receive a mesh
with `nothing` fields.

# Fields and conventions
- `p :: (np, 2)` — vertex coordinates.
- `t :: (nt, 3)` — triangles as vertex indices, counterclockwise.
- `f :: (nf, 4)` — faces: `f[i, 1:2]` are the endpoint vertices, `f[i, 3]`
  the element to the left, `f[i, 4]` the element to the right **or**
  `-k` when face `i` lies on boundary `k` (see [`boundary_names`](@ref)).
  Interior faces are listed first, then boundary faces grouped by tag.
- `t2f :: (nt, 3)` — element-to-face map (unsigned): local face `j` of
  element `it` is `t2f[it, j]`.
- `t2o :: (nt, 3)` — orientation code of each local face w.r.t. the face's
  stored traversal (`1` matching, `2` reversed in 2D; up to 6 codes on
  tetrahedra), indexing `master.perm[:, s, o]`. Replaces the former sign of
  `t2f`; meshes constructed with a signed `t2f` and no `t2o` are migrated
  automatically for one release.
- `fcurved :: (nf,)`, `tcurved :: (nt,)` — flags marking faces/elements that
  touch a curved boundary (their high-order nodes are projected during
  [`createnodes`](@ref)).
- `porder` — polynomial order of the elements.
- `plocal :: (npl, 3)`, `tlocal` — master-element node positions
  (barycentric) and their triangulation, `npl = (porder+1)(porder+2)/2`.
- `dgnodes :: (npl, 2, nt)` — coordinates of the high-order DG nodes of each
  element (isoparametric on curved elements).
- `pcg`, `tcg` — deduplicated continuous (CG) node coordinates and
  connectivity, filled by [`cgmesh`](@ref).
- `elcon :: (porder+1, 3, nt)` — element-to-global trace-node connectivity
  used by HDG: global numbering of the face nodes of each local face, already
  orientation-corrected via the sign of `t2f`.
- `f2f :: (nf, 5)` — face-to-face connectivity ([`mkf2f`](@ref)), used by the
  block-Jacobi preconditioner.
- `boundary_names :: Vector{Symbol}` — names of the boundary tags, in tag
  order (empty when the generator did not attach names).
"""
struct Mesh{Dim, P, T, F, TF, TO, FC, TC, PO, PL, TL, DG, PCG, TCG, ELC, FTF, BN}
                     p :: P
                     t :: T
                     f :: F
                   t2f :: TF
                   t2o :: TO
               fcurved :: FC
               tcurved :: TC
                porder :: PO
                plocal :: PL
                tlocal :: TL
               dgnodes :: DG
                   pcg :: PCG
                   tcg :: TCG
                 elcon :: ELC
                   f2f :: FTF
        boundary_names :: BN

    function Mesh{Dim}(p::P, t::T, f::F, t2f::TF, t2o::TO, fcurved::FC, tcurved::TC,
                       porder::PO, plocal::PL, tlocal::TL, dgnodes::DG, pcg::PCG,
                       tcg::TCG, elcon::ELC, f2f::FTF, boundary_names::BN) where
                      {Dim, P, T, F, TF, TO, FC, TC, PO, PL, TL, DG, PCG, TCG, ELC, FTF, BN}
        return new{Dim, P, T, F, TF, TO, FC, TC, PO, PL, TL, DG, PCG, TCG, ELC, FTF, BN}(
            p, t, f, t2f, t2o, fcurved, tcurved, porder, plocal, tlocal,
            dgnodes, pcg, tcg, elcon, f2f, boundary_names)
    end
end

"""
    ndims(mesh) -> Int

Spatial dimension of the mesh (the `Dim` type parameter): 2 for triangles,
3 for tetrahedra.
"""
Base.ndims(::Mesh{Dim}) where {Dim} = Dim

function Mesh(; p, t, f=nothing, t2f=nothing, t2o=nothing, fcurved=nothing, tcurved=nothing, porder, plocal, tlocal, dgnodes=nothing, pcg=nothing, tcg=nothing, elcon=nothing, f2f=nothing, boundary_names=Symbol[])
    # migration path (one release): a signed t2f with no t2o carries the
    # orientation in its sign — split it into the explicit representation
    if t2o === nothing && t2f isa AbstractMatrix && any(<(0), t2f)
        t2o = ifelse.(t2f .< 0, 2, 1)
        t2f = abs.(t2f)
    elseif t2o === nothing && t2f isa AbstractMatrix
        t2o = fill(1, size(t2f))
    end
    return Mesh{size(p, 2)}(p, t, f, t2f, t2o, fcurved, tcurved, porder, plocal, tlocal, dgnodes, pcg, tcg, elcon, f2f, boundary_names)
end

function Mesh(mesh::Mesh; dgnodes=nothing, pcg=nothing, tcg=nothing, elcon=nothing, f2f=nothing, boundary_names=mesh.boundary_names)
    return Mesh(; mesh.p, mesh.t, mesh.f, mesh.t2f, mesh.t2o, mesh.fcurved, mesh.tcurved, mesh.porder, mesh.plocal, mesh.tlocal, dgnodes, pcg, tcg, elcon, f2f, boundary_names)
end

"""
    boundary_names(mesh) -> Vector{Symbol}

Names of the mesh's boundary segments, in boundary-tag order (tag `k` is
stored as `-k` in `mesh.f[:, 4]`). Empty when the generator did not attach
names.
"""
boundary_names(mesh::Mesh) = mesh.boundary_names

# Converts barycentric coordinates (λ) to Cartesian coordinates using vertices v₁, v₂, v₃
# λ contains the coordinates [λ₂, λ₃], with λ₁ = 1-λ₂-λ₃ implied
function barycentric_to_cartesian(λ, v₁, v₂, v₃)
    # Create transformation matrix from barycentric to Cartesian
    # T maps the barycentric space to the triangle in Cartesian space
    T = hcat(v₂ .- v₁, v₃ .- v₁)
    # Apply transformation and shift by v₁
    return T * λ .+ v₁
end

# Creates a function that returns the derivative of f using forward automatic differentiation
autodiff(f) = x -> ForwardDiff.derivative(f, x)

# Projects a point x₀ onto the boundary defined by distance_function
# s is an initial guess for the distance to the boundary
function project_to_boundary(distance_function, x₀, s=0)
    # Calculate gradient of the distance function at x₀
    grad = ForwardDiff.gradient(distance_function, x₀)
    
    # If gradient is zero, x₀ is already at a critical point
    if iszero(grad)
        return x₀
    else
        # Normalize gradient to get direction vector
        grad_norm = grad / norm(grad)
        
        # Define a 1D function along the gradient direction
        fd_linedirection(s) = distance_function(x₀ .+ s .* grad_norm)
        
        # Use Newton-Raphson to find the value of s where fd_linedirection(s) = 0
        # (i.e., find where the point lands on the boundary)
        s = newton_raphson(fd_linedirection, autodiff(fd_linedirection), s)
        
        # Return the projected point
        return x₀ .+ s .* grad_norm
    end
end

# Checks if a barycentric coordinate λ represents a vertex
# λ = [λ₁, λ₂, λ₃] where λ₁ = 1-λ₂-λ₃ is implied
function isvertex(λ)
    λ₁ = λ[2]
    λ₂ = λ[3]
    # A vertex occurs when coordinates are binary (0 or 1)
    return (λ₁ == 0 || λ₁ == 1) && (λ₂ == 0 || λ₂ == 1)
end

# Checks if a barycentric coordinate λ lies on an edge
function isedge(λ)
    # An edge occurs when any barycentric coordinate is zero
    return any(λ .== 0)
end

# Determines which edge the barycentric coordinate λ lies on
function edge_number(λ)
    if λ[3] == 0
        return 1  # Edge between vertices 1 and 2
    elseif λ[2] == 0
        return 3  # Edge between vertices 3 and 1
    else
        return 2  # Edge between vertices 2 and 3
    end
end

# Checks if a point (in barycentric coordinates λ) lies on a curved edge
# vn₁, vn₂, vn₃ are vertex indices, it is the triangle index
function iscurvededge(λ, mesh, vn₁, vn₂, vn₃, it)
    # Get all curved faces from the mesh
    # mesh.fcurved contains indices of curved faces
    all_curved_faces = mesh.f[mesh.fcurved, :]
    
    # Find the row where the third column equals it
    row_number = findfirst(x -> x == it, all_curved_faces[:, 3])
    curved_face = all_curved_faces[row_number, :]
    
    # Determine which edge we're on based on barycentric coordinates
    Eₙ = edge_number(λ)
    
    # Check if the vertices of this edge match the curved face definition
    if Eₙ == 1
        return (vn₁ == curved_face[1] && vn₂ == curved_face[2]) || (vn₁ == curved_face[2] && vn₂ == curved_face[1])
    elseif Eₙ == 2
        return (vn₂ == curved_face[1] && vn₃ == curved_face[2]) || (vn₂ == curved_face[2] && vn₃ == curved_face[1])
    else
        return (vn₃ == curved_face[1] && vn₁ == curved_face[2]) || (vn₃ == curved_face[2] && vn₁ == curved_face[1])
    end
end

# Checks if a point lies on a curved boundary
# In mesh.f, negative values in the 4th column indicate boundaries
function iscurvedboundary(λ, mesh, vn₁, vn₂, vn₃, it)
    Eₙ = edge_number(λ)
    
    # Find where boundary definitions start in mesh.f (indicated by negative values in 4th column)
    i_bnd = findfirst(x -> x < 0, mesh.f[:, 4])
    
    # Get all curved boundaries
    all_curved_boundaries = mesh.f[i_bnd:end, :]
    
    # Find the row for this triangle
    row_number = findfirst(x -> x == it, all_curved_boundaries[:, 3])
    
    if row_number === nothing
        return false
    elseif Eₙ == 1
        # Check if edge 1 (between vertices 1 and 2) matches the boundary definition
        return (vn₁ == all_curved_boundaries[row_number, 1] && vn₂ == all_curved_boundaries[row_number, 2]) || 
               (vn₁ == all_curved_boundaries[row_number, 2] && vn₂ == all_curved_boundaries[row_number, 1])
    elseif Eₙ == 2
        # Check if edge 2 (between vertices 2 and 3) matches the boundary definition
        return (vn₂ == all_curved_boundaries[row_number, 1] && vn₃ == all_curved_boundaries[row_number, 2]) || 
               (vn₂ == all_curved_boundaries[row_number, 2] && vn₃ == all_curved_boundaries[row_number, 1])
    else
        # Check if edge 3 (between vertices 3 and 1) matches the boundary definition
        return (vn₃ == all_curved_boundaries[row_number, 1] && vn₁ == all_curved_boundaries[row_number, 2]) || 
               (vn₃ == all_curved_boundaries[row_number, 2] && vn₁ == all_curved_boundaries[row_number, 1])
    end
end

# Gets the boundary number for a specific triangle
# The boundary number is stored as a negative value in mesh.f[:, 4]
function get_boundary_number(mesh, it)
    i_bnd = findfirst(x -> x < 0, mesh.f[:, 4])
    all_curved_boundaries = mesh.f[i_bnd:end, :]
    row_number = findfirst(x -> x == it, all_curved_boundaries[:, 3])
    # Return the absolute value of the boundary number
    return -all_curved_boundaries[row_number, 4]
end

# Projects vertices of a mesh onto their corresponding boundaries
# distance_functions: array of functions defining the signed distance to each boundary
function project_vertex_to_boundary!(mesh::Mesh, distance_functions::Union{Nothing, Vector})
    if distance_functions !== nothing
        # Find the start of boundary definitions in mesh.f
        i_bnd = findfirst(x -> x < 0, mesh.f[:, 4])
        
        # Count number of unique boundary curves
        n_curves = length(Set(mesh.f[i_bnd:end, 4]))
        all_curved_faces = mesh.f[i_bnd:end, :]
        
        for i in 1:n_curves
            fd = distance_functions[i]
            
            # Get faces associated with boundary i
            curved_faces = all_curved_faces[all_curved_faces[:, 4] .== -i, :]
            
            # Collect unique nodes on this boundary
            unique_curved_nodes = Dict{Int, Nothing}()
            for node in curved_faces[:, 1:2]
                unique_curved_nodes[node] = nothing
            end
            
            # Project each boundary node onto the boundary curve
            for node in keys(unique_curved_nodes)
                node_coords = mesh.p[node, :]
                mesh.p[node, :] .= project_to_boundary(fd, node_coords, 0)
            end
        end
    end
end

"""
    orient_perm(nps, Val(2))          -> (nps, 2)  permutation table
    orient_perm(face_plocal, Val(3))  -> (nps, 6)  permutation table

Trace-node permutations of one face under each orientation code: an element
seeing a face with orientation `o` couples its `k`-th local trace node (its
own canonical face traversal) to the face's global trace node
`op[k, o]`. In 2D this is the identity and the reversal; in 3D it is built
constructively from the triangle face element's node barycentrics: `op[k, o]`
is the canonical face node `m` whose coordinates permuted by the `o`-th
triangle symmetry match node `k` (`face_plocal[m, σ_o] == face_plocal[k, :]`).
"""
orient_perm(nps::Integer, ::Val{2}) = hcat(1:nps, nps:-1:1)

function orient_perm(face_plocal::AbstractMatrix, ::Val{3})
    nps = size(face_plocal, 1)
    orients = orientation_permutations(Val(3))
    op = zeros(Int, nps, length(orients))
    for (o, σ) in enumerate(orients), k in 1:nps
        m = findfirst(1:nps) do i
            all(d -> abs(face_plocal[i, σ[d]] - face_plocal[k, d]) < 1e-8, 1:3)
        end
        m === nothing &&
            error("face node set is not invariant under the triangle symmetry group (orientation $o, node $k)")
        op[k, o] = m
    end
    return op
end

"""
    mkelcon(t2f, t2o, porder)

Element-to-global trace-node connectivity `elcon (porder+1, 3, nt)` from the
element-to-face map `t2f` and orientation codes `t2o`: face `f` owns trace
nodes `(f-1)*(porder+1)+1 : f*(porder+1)` in the face's canonical order, and
each element traverses them through `orient_perm[:, t2o[it, s]]`.
"""
function mkelcon(t2f, t2o, porder)
    nps = porder + 1
    nt = size(t2f, 1)
    op = orient_perm(nps, Val(2))
    elcon = zeros(Int, nps, 3, nt)
    for it in 1:nt
        for iface in axes(t2f, 2)
            face = t2f[it, iface]
            base = (face - 1) * nps
            elcon[:, iface, it] .= base .+ op[:, t2o[it, iface]]
        end
    end
    return elcon
end

"""
    mkelcon(t2f, t2o, porder, face_plocal) -> elcon (nps, 4, nt)

3D (tetrahedral) trace connectivity: face `f` owns the `nps =
(porder+1)(porder+2)/2` trace nodes `(f-1)nps+1 : f*nps` in the face's
canonical node order (`face_plocal`, the triangle face element's nodes), and
each element couples through [`orient_perm`](@ref) at its orientation code.
"""
function mkelcon(t2f, t2o, porder, face_plocal::AbstractMatrix)
    nps = size(face_plocal, 1)
    nt = size(t2f, 1)
    op = orient_perm(face_plocal, Val(3))
    elcon = zeros(Int, nps, 4, nt)
    for it in 1:nt
        for iface in axes(t2f, 2)
            base = (t2f[it, iface] - 1) * nps
            elcon[:, iface, it] .= base .+ op[:, t2o[it, iface]]
        end
    end
    return elcon
end

"""
    straight_dgnodes(p, t, plocal) -> dgnodes (npl, 2, nt)

Map the master-element nodes `plocal` (barycentric, `(npl, 3)`) affinely into
every triangle of `(p, t)`. This is the straight-element part of
[`createnodes`](@ref); use it when the high-order nodes will be transformed
analytically afterwards (e.g. [`mkmesh_trefftz`](@ref)'s conformal maps).
"""
function straight_dgnodes(p, t, plocal)
    npl = size(plocal, 1)
    nt = size(t, 1)
    dgnodes = zeros(eltype(p), npl, 2, nt)
    for it in 1:nt
        v₁ = p[t[it, 1], :]
        v₂ = p[t[it, 2], :]
        v₃ = p[t[it, 3], :]
        for ipl in 1:npl
            dgnodes[ipl, :, it] .= barycentric_to_cartesian(plocal[ipl, 2:3], v₁, v₂, v₃)
        end
    end
    return dgnodes
end

"""
    createnodes(mesh, fd=nothing) -> Mesh

Compute the high-order DG node coordinates `dgnodes (npl, 2, nt)` of `mesh`
by mapping the master-element nodes (`mesh.plocal`) into each triangle. When
`fd` is a vector of signed-distance functions (one per boundary tag), nodes
on curved boundary edges are projected onto the true boundary, making those
elements isoparametric.

Also fills the HDG trace connectivity `elcon` and the face-to-face map `f2f`
when the mesh already has face connectivity ([`mkt2f`](@ref)). Returns a new
[`Mesh`](@ref) with the extra fields populated.
"""
function createnodes(mesh, fd=nothing)
    npl = size(mesh.plocal, 1)
    nt = size(mesh.t, 1)
    
    # First, project mesh vertices onto boundaries if needed
    project_vertex_to_boundary!(mesh, fd)
    dgnodes = zeros(npl, 2, nt)
    
    for it in axes(dgnodes, 3)
        # Get vertex indices for this triangle
        vn₁ = mesh.t[it, 1]
        vn₂ = mesh.t[it, 2]
        vn₃ = mesh.t[it, 3]
        
        # Get vertex coordinates
        v₁ = mesh.p[vn₁, :]
        v₂ = mesh.p[vn₂, :]
        v₃ = mesh.p[vn₃, :]
        
        # Check if this triangle has a curved edge
        iscurved_triangle = mesh.tcurved !== nothing && mesh.tcurved[it]
        
        # Loop through each local point within the triangle
        for ipl in axes(dgnodes, 1)
            # Get barycentric coordinates of this point
            λ = mesh.plocal[ipl, :]
            
            # Convert barycentric to Cartesian coordinates
            x = barycentric_to_cartesian(λ[2:3], v₁, v₂, v₃)
            
            # Store these coordinates in dgnodes
            dgnodes[ipl, :, it] .= x
            
            # Special handling for points on curved boundaries:
            # If all these conditions are met:
            # 1. We have distance functions
            # 2. This triangle has a curved edge
            # 3. The point is not a vertex
            # 4. The point is on an edge
            # 5. That edge is part of a curved boundary
            if fd !== nothing && iscurved_triangle && !isvertex(λ) && isedge(λ) && iscurvedboundary(λ, mesh, vn₁, vn₂, vn₃, it)
                # Get the boundary number for this curved edge
                fdn = get_boundary_number(mesh, it)
                
                # Project the point onto the curved boundary using the appropriate distance function
                x = project_to_boundary(fd[fdn], x, 0)
                
                # Update the node coordinates with the projected position
                dgnodes[ipl, :, it] .= x
            end
        end
    end

    elcon = mkelcon(mesh.t2f, mesh.t2o, mesh.porder)
    f2f = mkf2f(mesh.f, mesh.t2f)
    
    # Create and return a new mesh with the same structure but using the computed high-order nodes
    return Mesh(mesh; dgnodes, elcon, f2f)
end

"""
    unique_with_inverse(arr)

Find unique rows in a matrix and return inverse mapping.
Similar to `np.unique(arr, return_inverse=True, axis=0)` in NumPy.
"""
function unique_with_inverse(arr::Matrix{Int})
    # Create dictionary for fast lookup
    row_dict = Dict{Tuple{Int,Int}, Int}()
    unique_rows = Matrix{Int}(undef, 0, 2)
    
    # Find unique rows
    for i in 1:size(arr, 1)
        key = (arr[i,1], arr[i,2])
        if !haskey(row_dict, key)
            unique_rows = vcat(unique_rows, arr[i:i,:])
            row_dict[key] = size(unique_rows, 1)
        end
    end
    
    # Create inverse mapping
    pairj = [row_dict[(arr[i,1], arr[i,2])] for i in 1:size(arr, 1)]
    
    return unique_rows, pairj
end

"""
    uniref(p, t, nref=1)

Uniformly refine a simplicial mesh: each triangle into 4 (edge midpoints),
each tetrahedron into 8 (Bey's red refinement — 4 corner tets plus the
interior octahedron split around its **shortest diagonal** for shape
regularity; Bey, Computing 55, 1995).

# Arguments
- `p::Matrix{T}`: Nodes as a matrix of size (np, dim)
- `t::Matrix{Int}`: Triangulation as a matrix of size (nt, dim+1)
- `nref::Int=1`: Number of uniform refinements

# Returns
- `p::Matrix{T}`: Refined nodes
- `t::Matrix{Int}`: Refined triangulation (positive orientation preserved)
"""
function uniref(p::Matrix{T}, t::Matrix{Int}, nref::Int=1) where T <: AbstractFloat
    for _ in 1:nref
        p, t = size(t, 2) == 3 ? _uniref_tri(p, t) : _uniref_tet(p, t)
    end
    return p, t
end

function _uniref_tri(p::Matrix{T}, t::Matrix{Int}) where T
    n = size(p, 1)
    nt = size(t, 1)

    # Extract all edges from triangulation
    pair = zeros(Int, 3*nt, 2)
    for i in 1:nt
        pair[i,:] = [t[i,1], t[i,2]]
        pair[nt+i,:] = [t[i,1], t[i,3]]
        pair[2*nt+i,:] = [t[i,2], t[i,3]]
    end

    # Sort each row
    for i in 1:size(pair, 1)
        if pair[i,1] > pair[i,2]
            pair[i,1], pair[i,2] = pair[i,2], pair[i,1]
        end
    end

    # Find unique edges and their inverse indices
    unique_pair, pairj = unique_with_inverse(pair)

    # Calculate midpoints
    pmid = zeros(T, size(unique_pair, 1), size(p, 2))
    for i in 1:size(unique_pair, 1)
        pmid[i,:] = (p[unique_pair[i,1],:] + p[unique_pair[i,2],:]) ./ 2
    end

    # Create new triangulation
    t1 = t[:,1]
    t2 = t[:,2]
    t3 = t[:,3]

    t12 = pairj[1:nt] .+ n
    t13 = pairj[(nt+1):(2*nt)] .+ n
    t23 = pairj[(2*nt+1):(3*nt)] .+ n

    # Construct new triangulation
    t_new = zeros(Int, 4*nt, 3)
    for i in 1:nt
        t_new[i,:] = [t1[i], t12[i], t13[i]]
        t_new[nt+i,:] = [t12[i], t23[i], t13[i]]
        t_new[2*nt+i,:] = [t2[i], t23[i], t12[i]]
        t_new[3*nt+i,:] = [t3[i], t13[i], t23[i]]
    end

    return vcat(p, pmid), t_new
end

# Bey red refinement 1 → 8: four corner tets plus the interior octahedron of
# edge midpoints, split into four tets around whichever of its three
# diagonals is shortest (the choice that bounds the shape-regularity of the
# refined family).
function _uniref_tet(p::Matrix{T}, t::Matrix{Int}) where T
    n = size(p, 1)
    nt = size(t, 1)
    edges = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

    pair = zeros(Int, 6*nt, 2)
    for i in 1:nt, (k, (a, b)) in enumerate(edges)
        va, vb = t[i, a], t[i, b]
        pair[(k - 1) * nt + i, 1] = min(va, vb)
        pair[(k - 1) * nt + i, 2] = max(va, vb)
    end
    unique_pair, pairj = unique_with_inverse(pair)

    pmid = zeros(T, size(unique_pair, 1), size(p, 2))
    for i in 1:size(unique_pair, 1)
        pmid[i, :] = (p[unique_pair[i, 1], :] + p[unique_pair[i, 2], :]) ./ 2
    end
    pnew = vcat(p, pmid)

    # equatorial cycle of the octahedron for each choice of diagonal, as
    # indices into (m12, m13, m14, m23, m24, m34); opposite midpoint pairs
    # (the diagonals) are (m12,m34), (m13,m24), (m14,m23)
    diags = ((1, 6), (2, 5), (3, 4))
    cycles = ((2, 3, 5, 4), (1, 3, 6, 4), (1, 2, 6, 5))

    t_new = zeros(Int, 8*nt, 4)
    row = 0
    for i in 1:nt
        m = ntuple(k -> n + pairj[(k - 1) * nt + i], 6)   # m12 m13 m14 m23 m24 m34
        v = ntuple(k -> t[i, k], 4)

        # corner tets
        corners = ((v[1], m[1], m[2], m[3]),
                   (m[1], v[2], m[4], m[5]),
                   (m[2], m[4], v[3], m[6]),
                   (m[3], m[5], m[6], v[4]))

        # shortest octahedron diagonal
        len2(a, b) = sum(k -> (pnew[a, k] - pnew[b, k])^2, 1:size(pnew, 2))
        kbest = argmin([len2(m[d[1]], m[d[2]]) for d in diags])
        a, b = m[diags[kbest][1]], m[diags[kbest][2]]
        c = ntuple(j -> m[cycles[kbest][j]], 4)
        octs = ((a, c[1], c[2], b), (a, c[2], c[3], b),
                (a, c[3], c[4], b), (a, c[4], c[1], b))

        for tt in (corners..., octs...)
            row += 1
            t_new[row, 1] = tt[1]; t_new[row, 2] = tt[2]
            t_new[row, 3] = tt[3]; t_new[row, 4] = tt[4]
        end
    end

    # restore positive orientation (children of the octahedron split may come
    # out either-handed depending on the diagonal chosen)
    vols = simpvol(pnew, t_new)
    for i in findall(<(0), vols)
        t_new[i, 3], t_new[i, 4] = t_new[i, 4], t_new[i, 3]
    end

    return pnew, t_new
end

"""
    mkf2f(f, t2f)

Create face to face connectivity.

# Arguments
- `f`: Face to element connectivity
- `t2f`: Element to face connectivity

# Returns
- `f2f::Matrix{Int}`: Face to face connectivity
"""
function mkf2f(f, t2f)
    t2f_abs = abs.(t2f)
    nf = size(f, 1)  # number of faces
    nfe = size(t2f, 2)  # number of faces per element
    nbf = 2*nfe-1  # number of neighboring faces
    f2f = zeros(Int, nf, nbf)
    
    # This operation is highly parallelizable
    for i in 1:nf  # Julia uses 1-based indexing
        fi = f[i, end-1:end]  # obtain two elements sharing the same face i

        if fi[2] >= 0  # face i is an interior face
            kf = t2f_abs[fi, :]  # get neighboring faces
            # Find the index of face i in the elements
            i1 = findfirst(==(i), kf[1, :])
            i2 = findfirst(==(i), kf[2, :])
            
            # The first block
            k = 1  # Start with 1 in Julia
            f2f[i, k] = i  # Store face number (already 1-indexed in Julia)
            
            # Process first element's faces
            for is_ in 1:nfe
                if is_ != i1
                    k += 1
                    j = kf[1, is_]
                    f2f[i, k] = j
                end
            end
            
            # Process second element's faces
            for is_ in 1:nfe
                if is_ != i2
                    k += 1
                    j = kf[2, is_]
                    f2f[i, k] = j
                end
            end
        
        else  # face i is a boundary face
            kf = t2f_abs[fi[1], :]  # obtain neighboring faces
            i1 = findfirst(==(i), kf)  # obtain the index of face i in the element
            
            # The first block
            k = 1
            f2f[i, k] = i  # Store face number
            
            # Process element's faces
            for is_ in 1:nfe
                if is_ != i1
                    k += 1
                    j = kf[is_]
                    f2f[i, k] = j
                end
            end
        end
    end
    
    return f2f
end