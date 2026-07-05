using LinearAlgebra
using TwoDG.Meshes: Mesh, face_vertices, orientation_permutations, norient

"""
    ReferenceElement(porder; pgauss=max(4porder, 1), nodetype=0)
    ReferenceElement(plocal, porder; pgauss=max(4porder, 1))
    ReferenceElement(mesh::Mesh, pgauss=nothing)

Reference (master) simplex of polynomial order `porder` in `Dim` dimensions:
tabulated shape functions, quadrature rules, and the local node orderings the
solvers need. The face of the element is itself the `Dim-1`-dimensional
reference simplex, stored whole in the `face` field (the triangle's face is
the 1D segment; the tetrahedron's face is the full triangle element), so face
quadrature, face shape functions, and the HDG trace basis come from one
recursive structure. The element is mesh-independent; the `Mesh` convenience
constructor just reads `mesh.porder`/`mesh.plocal` so the element's nodes
match the mesh's `dgnodes`.

`pgauss` is the polynomial degree integrated exactly by the quadrature rules;
`nodetype` selects the node distribution (`0` uniform, `1`/`2` extended
Chebyshev, see [`localpnts`](@ref)); `plocal` may be passed directly to use a
custom node set.

# Fields and conventions (`nv = Dim + 1` vertices/faces, `npf` nodes per face,
`norient` face orientations: 1 in 1D, 2 in 2D, 6 in 3D)
- `porder` — polynomial order; `npl` nodes (`(porder+1)(porder+2)/2` for the
  triangle, `(porder+1)(porder+2)(porder+3)/6` for the tetrahedron).
- `plocal :: (npl, nv)` — node positions in barycentric coordinates.
- `corner :: Vector{Int}` — indices of the `nv` vertex nodes in `plocal`.
- `perm :: (npf, nv, norient)` — face-node orderings: `perm[:, j, o]` lists
  the volume nodes on local face `j` matching the face's canonical traversal
  under orientation `o` (`o = 1` canonical, `o = 2` reversed in 2D; the 6
  triangle symmetries in 3D). Used with `mesh.t2o` when a neighboring element
  sees the shared face in a different orientation.
- `gpts :: (ng, Dim)`, `gwgh :: (ng,)` — volume quadrature points/weights.
- `shap :: (npl, Dim+1, ng)` — shape functions (`[:, 1, :]`) and their
  reference-coordinate derivatives (`[:, 1+d, :]` for direction `d`) at the
  volume quadrature points.
- `mass :: (npl, npl)` — reference-element mass matrix.
- `conv :: (npl, npl, Dim)` — reference convection matrices
  (``∫ φᵢ ∂φⱼ/∂ξ_d``).
- `face` — the `Dim-1`-dimensional [`ReferenceElement`](@ref) of the faces
  (`nothing` for the 1D segment), built with the matching node distribution
  and quadrature degree.

# Deprecated property aliases (2D)
The pre-`Dim` 1D face tables remain readable as properties for one release:
`master.sh1d == master.face.shap`, `master.ma1d == master.face.mass`,
`master.gw1d == master.face.gwgh`, `master.gp1d == vec(master.face.gpts)`,
`master.ploc1d == master.face.plocal`.
"""
struct ReferenceElement{Dim, T <: AbstractFloat, F}   # F: the face ReferenceElement, or Nothing in 1D
    porder :: Int
    plocal :: Matrix{T}
    corner :: Vector{Int}
      perm :: Array{Int, 3}
      gpts :: Matrix{T}
      gwgh :: Vector{T}
      shap :: Array{T, 3}
      mass :: Matrix{T}
      conv :: Array{T, 3}
      face :: F
end

# 1D face tables of the pre-Dim ReferenceElement, kept as property aliases for
# one release (NEWS.md): they are exactly the face element's tables.
@inline function Base.getproperty(m::ReferenceElement, s::Symbol)
    s === :sh1d   && return getfield(m, :face).shap
    s === :ma1d   && return getfield(m, :face).mass
    s === :gw1d   && return getfield(m, :face).gwgh
    s === :gp1d   && return vec(getfield(m, :face).gpts)
    s === :ploc1d && return getfield(m, :face).plocal
    return getfield(m, s)
end

Base.propertynames(m::ReferenceElement{2}) =
    (fieldnames(ReferenceElement)..., :sh1d, :ma1d, :gw1d, :gp1d, :ploc1d)

Base.ndims(::ReferenceElement{Dim}) where {Dim} = Dim
Base.eltype(::ReferenceElement{Dim, T}) where {Dim, T} = T

# the shared face-orientation vocabulary (face_vertices,
# orientation_permutations, norient, face_orientation) lives in
# `Meshes/orientation.jl` — one definition serving mkt2f/t2o, the reference
# element's perm, and the HDG trace numbering

"""
    build_face_perm(::Val{Dim}, plocal, face_plocal) -> perm (npf, Dim+1, norient)

Constructive face-node permutation table: for each local face `j` and
orientation `o`, `perm[:, j, o]` lists the volume-node indices whose
barycentric coordinates (restricted to the face's vertices) match the face
element's nodes `face_plocal` under the `o`-th symmetry of the face simplex.
No hand-tabulated index magic — works for every `porder`, provided the volume
node set restricted to each face equals the face element's node set (true for
the symmetrized [`localpnts`](@ref)/`localpnts3d` distributions; asserted).
"""
function build_face_perm(::Val{Dim}, plocal::AbstractMatrix{T},
                         face_plocal::AbstractMatrix) where {Dim, T}
    npf = size(face_plocal, 1)
    orients = orientation_permutations(Val(Dim))
    perm = zeros(Int, npf, Dim + 1, length(orients))
    for j in 1:(Dim + 1)
        onface = findall(<(1e-6), @view plocal[:, j])
        length(onface) == npf ||
            error("face $j has $(length(onface)) nodes; the face element has $npf — node set is not face-compatible")
        fv = face_vertices(Val(Dim), j)
        facebary = plocal[onface, collect(fv)]           # (npf, Dim)
        for (o, σ) in enumerate(orients)
            for k in 1:npf
                target = ntuple(d -> face_plocal[k, σ[d]], Val(Dim))
                idx = findfirst(1:npf) do m
                    all(d -> abs(facebary[m, d] - target[d]) < 1e-8, 1:Dim)
                end
                idx === nothing &&
                    error("face $j node set is not invariant under the face-simplex symmetry group (orientation $o, node $k)")
                perm[k, j, o] = onface[idx]
            end
        end
    end
    return perm
end

# --- 1D segment element (the face of the triangle) ---------------------------
function _segment_element(ploc1d::Matrix{T}, porder::Integer, gp1d::Vector{T},
                          gw1d::Vector{T}, sh1d::Array{<:Real, 3}) where {T}
    np = porder + 1
    corner = [findfirst(<(1e-6), @view ploc1d[:, 2]),   # x = 0 vertex
              findfirst(>(1 - 1e-6), @view ploc1d[:, 2])]
    # faces of the segment are its endpoints: face j is where barycentric
    # coordinate j vanishes (face 1 at x = 1, face 2 at x = 0); one node, one
    # orientation.
    perm = reshape([corner[2], corner[1]], 1, 2, 1)
    sh1dT = Array{T, 3}(sh1d)
    mass = sh1dT[:, 1, :] * Diagonal(gw1d) * sh1dT[:, 1, :]'
    conv = Array{T, 3}(undef, np, np, 1)
    conv[:, :, 1] .= sh1dT[:, 1, :] * Diagonal(gw1d) * sh1dT[:, 2, :]'
    return ReferenceElement{1, T, Nothing}(Int(porder), ploc1d, corner, perm,
                                           reshape(gp1d, :, 1), gw1d, sh1dT,
                                           mass, conv, nothing)
end

# route by the barycentric width of plocal: 3 columns = triangle, 4 = tet
ReferenceElement(plocal::AbstractMatrix{<:Real}, porder::Integer;
                 pgauss::Integer=max(4porder, 1)) =
    ReferenceElement(plocal, porder, Val(size(plocal, 2) - 1); pgauss)

function ReferenceElement(plocal::AbstractMatrix{<:Real}, porder::Integer, ::Val{2};
                          pgauss::Integer=max(4porder, 1))
    T = float(eltype(plocal))
    plocal = Matrix{T}(plocal)
    npl = size(plocal, 1)
    npl == (porder + 1) * (porder + 2) ÷ 2 ||
        throw(ArgumentError("plocal has $npl nodes; order $porder needs $((porder + 1) * (porder + 2) ÷ 2)"))

    # vertex nodes: the ones with a barycentric coordinate equal to 1
    corner = map(1:3) do i
        c = findfirst(>(1 - 1e-6), @view plocal[:, i])
        c === nothing && throw(ArgumentError("plocal has no vertex node for corner $i"))
        c
    end

    # face j is the edge where barycentric coordinate j vanishes; traverse it
    # counterclockwise (orientation 1), and reversed (orientation 2) for the
    # neighboring element that sees the shared face with opposite orientation.
    # (Equivalent to `build_face_perm` for symmetric node sets, but valid for
    # any custom `plocal`; the equivalence is asserted in the test suite.)
    perm = zeros(Int, porder + 1, 3, 2)
    aux = (1, 2, 3, 1, 2)
    ploc1d = Matrix{T}(undef, porder + 1, 2)
    for i in 1:3
        onface = findall(<(1e-6), @view plocal[:, i])
        order = sortperm(plocal[onface, aux[i + 2]])
        perm[:, i, 1] .= onface[order]
        if i == 3
            ploc1d .= plocal[onface[order], 1:2]
        end
    end
    perm[:, :, 2] .= reverse(@view(perm[:, :, 1]), dims=1)

    gp1d, gw1d = gaussquad1d(pgauss)   # 1D (face) quadrature
    gpts, gwgh = gaussquad2d(pgauss)   # 2D (volume) quadrature

    sh1d = shape1d(porder, ploc1d, Vector{T}(gp1d))
    shap = shape2d(porder, plocal, gpts)

    mass = shap[:, 1, :] * Diagonal(gwgh) * shap[:, 1, :]'
    conv = Array{T}(undef, npl, npl, 2)
    conv[:, :, 1] .= shap[:, 1, :] * Diagonal(gwgh) * shap[:, 2, :]'
    conv[:, :, 2] .= shap[:, 1, :] * Diagonal(gwgh) * shap[:, 3, :]'

    face = _segment_element(ploc1d, porder, Vector{T}(gp1d), Vector{T}(gw1d), sh1d)

    return ReferenceElement{2, T, typeof(face)}(Int(porder), plocal, corner, perm,
                                                Matrix{T}(gpts), Vector{T}(gwgh),
                                                Array{T, 3}(shap), mass, conv, face)
end

"""
    ReferenceElement(porder; dim=2, nodetype=0, pgauss=max(4porder, 1))

Reference simplex of order `porder`: the triangle for `dim = 2`, the
tetrahedron for `dim = 3` (whose `face` field is the full triangle element).
"""
function ReferenceElement(porder::Integer; dim::Integer=2, nodetype::Integer=0,
                          pgauss::Integer=max(4porder, 1))
    dim == 2 && return ReferenceElement(first(localpnts(porder, nodetype)), porder; pgauss)
    dim == 3 && return ReferenceElement(localpnts3d(porder, nodetype), porder; pgauss)
    throw(ArgumentError("dim must be 2 or 3, got $dim"))
end

ReferenceElement(mesh::Mesh, pgauss=nothing) =
    ReferenceElement(mesh.plocal, mesh.porder;
                     pgauss=pgauss === nothing ? max(4 * mesh.porder, 1) : pgauss)

# --- 3D tetrahedral element ---------------------------------------------------
# Dispatched from the generic constructor by the barycentric width of plocal
# (4 columns = tetrahedron). The face element is the *existing triangle*
# element built from the face-restricted node set — face quadrature, face
# shape functions, and the HDG trace basis in 3D come from the 2D code.
function ReferenceElement(plocal::AbstractMatrix{<:Real}, porder::Integer,
                          ::Val{3}; pgauss::Integer=max(4porder, 1))
    T = float(eltype(plocal))
    plocal = Matrix{T}(plocal)
    npl = size(plocal, 1)
    npl == (porder + 1) * (porder + 2) * (porder + 3) ÷ 6 ||
        throw(ArgumentError("plocal has $npl nodes; order $porder needs $((porder + 1) * (porder + 2) * (porder + 3) ÷ 6)"))

    corner = map(1:4) do i
        c = findfirst(>(1 - 1e-6), @view plocal[:, i])
        c === nothing && throw(ArgumentError("plocal has no vertex node for corner $i"))
        c
    end

    # the face element: the triangle carrying the restriction of the volume
    # node set to local face 1, in that face's own barycentric coordinates
    onface = findall(<(1e-6), @view plocal[:, 1])
    fv = face_vertices(Val(3), 1)
    face_plocal = plocal[onface, collect(fv)]
    face = ReferenceElement(face_plocal, porder; pgauss)

    # face-node orderings for all 6 relative orientations, built
    # constructively from the face-simplex symmetries (D7) — requires the
    # node set to be invariant under them (true for the symmetric localpnts3d
    # distributions; checked inside)
    perm = build_face_perm(Val(3), plocal, face.plocal)

    gpts, gwgh = gaussquad3d(pgauss)
    shap = shape3d(porder, plocal, gpts)

    mass = shap[:, 1, :] * Diagonal(gwgh) * shap[:, 1, :]'
    conv = Array{T}(undef, npl, npl, 3)
    for d in 1:3
        conv[:, :, d] .= shap[:, 1, :] * Diagonal(gwgh) * shap[:, 1 + d, :]'
    end

    return ReferenceElement{3, T, typeof(face)}(Int(porder), plocal, corner, perm,
                                                Matrix{T}(gpts), Vector{T}(gwgh),
                                                Array{T, 3}(shap), mass, conv, face)
end

"""
    shape3d(porder, plocal, pts)

Nodal shape functions and derivatives on the master tetrahedron
`[0,0,0]-[1,0,0]-[0,1,0]-[0,0,1]`: `nfs (npl, 4, npoints)` with values in
`[:, 1, :]` and the ξ/η/ζ derivatives in `[:, 2:4, :]`. Same
Vandermonde-solve pattern as [`shape2d`](@ref), on the [`koornwinder3d`](@ref)
basis.
"""
function shape3d(porder, plocal, pts)
    np = (porder + 1) * (porder + 2) * (porder + 3) ÷ 6
    npoints = size(pts, 1)

    W, _, _, _ = koornwinder3d(plocal[:, 2:4], porder)
    A = W \ I

    Λ, Λξ, Λη, Λζ = koornwinder3d(pts, porder)

    nfs = zeros(np, 4, npoints)
    nfs[:, 1, :] .= (Λ * A)'
    nfs[:, 2, :] .= (Λξ * A)'
    nfs[:, 3, :] .= (Λη * A)'
    nfs[:, 4, :] .= (Λζ * A)'

    return nfs
end

"""
    localpnts3d(porder, nodetype=0) -> plocal (npl, 4)

Node positions on the master tetrahedron in barycentric coordinates,
`npl = (porder+1)(porder+2)(porder+3)/6`. `nodetype = 0` places nodes on the
uniform barycentric lattice — which is invariant under the tetrahedron's
symmetry group and restricts to the uniform triangle nodes on every face,
the two properties `perm`/`t2o` depend on (asserted in the constructor).
Warp-and-blend nodes (Warburton 2006) are a planned optimization for high
`porder`.
"""
function localpnts3d(porder::Integer, nodetype::Integer=0)
    nodetype == 0 ||
        throw(ArgumentError("localpnts3d currently supports only the uniform node distribution (nodetype = 0)"))
    npl = (porder + 1) * (porder + 2) * (porder + 3) ÷ 6
    plocal = zeros(npl, 4)
    if porder == 0
        plocal[1, :] .= 0.25
        return plocal
    end
    m = 0
    for k in 0:porder, j in 0:(porder - k), i in 0:(porder - k - j)
        m += 1
        plocal[m, 2] = i / porder
        plocal[m, 3] = j / porder
        plocal[m, 4] = k / porder
        plocal[m, 1] = 1 - (i + j + k) / porder
    end
    return plocal
end

"""
    Master

Deprecated alias for [`ReferenceElement`](@ref); will be removed one release
after the rename (see NEWS.md).
"""
const Master = ReferenceElement

"""
uniformlocalpnts 2-d mesh generator for the master element.
[plocal,tlocal]=uniformlocalpnts(porder)

   plocal:    node positions (npl,3)
   tlocal:    triangle indices (nt,3)
   porder:    order of the complete polynomial 
              npl = (porder+1)*(porder+2)/2
 """
 function uniformlocalpnts(porder)
    plocal = zeros()
    n = porder + 1
    npl = (porder + 1) * (porder + 2) ÷ 2  # Total number of nodes for order porder

    plocal = zeros(npl, 3)  # Initialize array for barycentric coordinates
    xs = ys = range(0, 1, length=n)  # Create uniform distribution in each direction
    
    # Generate nodal points in barycentric coordinates
    # We're placing points on a regular grid and converting to barycentric coordinates
    i_start = 1
    for i in 1:n
        i_end = i_start + n - i
        # Set barycentric coordinates:
        # 2nd coordinate increases with row index
        plocal[i_start:i_end, 2] .= xs[1:n+1-i]
        # 3rd coordinate is constant in each row
        plocal[i_start:i_end, 3] .= ys[i]
        # 1st coordinate decreases to maintain sum(coords) = 1
        plocal[i_start:i_end, 1] .= xs[n+1-i:-1:1]
        i_start = i_end + 1
    end

    # Generate triangle connectivity based on the nodal distribution
    # This creates a triangulation of the reference element
    tlocal = zeros(Int, porder^2, 3)
    i_start_t = 1
    vertex_start = 1
    for i in 1:porder
        # Create the first set of triangles in this row (pointing up)
        i_end_t = i_start_t + porder - i
        tlocal[i_start_t:i_end_t, 1] .= vertex_start:vertex_start + porder - i
        tlocal[i_start_t:i_end_t, 2] .= vertex_start + 1:vertex_start + porder - i + 1
        tlocal[i_start_t:i_end_t, 3] .= vertex_start + porder - i + 2:vertex_start + 2porder - 2i + 2
        
        i_start_t = i_end_t + 1

        # Create the second set of triangles in this row (pointing down)
        # except for the last row which only has upward triangles
        if i_start_t < porder^2
            i_end_t = i_start_t + porder - i - 1
            vertex_start += 1
            tlocal[i_start_t:i_end_t, 1] .= vertex_start:vertex_start + porder - i - 1
            tlocal[i_start_t:i_end_t, 2] .= vertex_start + porder - i + 2:vertex_start + 2porder - 2i + 1
            tlocal[i_start_t:i_end_t, 3] .= vertex_start + porder - i + 1:vertex_start + 2porder - 2i
            i_start_t = i_end_t + 1
        end
        
        vertex_start += porder - i + 1
    end

    return plocal, tlocal
end

"""
shape1d calculates the nodal shapefunctions and its derivatives for
         the master 1d element [0,1]

Arguments:
- `porder`: polynomial order
- `plocal`: node positions (np,2) (np=porder+1)
- `pts`: coordinates of the points where the shape functions
         and derivatives are to be evaluated (npoints)

Returns:
- `nsf`: shape function and derivatives (np,2,npoints)
         nsf[:,1,:] shape functions
         nsf[:,2,:] shape functions derivatives w.r.t. x
"""
function shape1d(porder::Int, plocal::AbstractMatrix{T}, pts::AbstractVector{T}) where T <: Real
    f, fx = koornwinder1d(pts, porder)
    A, _ = koornwinder1d(@view(plocal[:,2]), porder)  # Using column 2 due to 1-based indexing
    
    # Solve linear systems efficiently using Julia's \ operator
    nf = A' \ f'
    nfx = A' \ fx'
    
    # Preallocate memory for the result
    nfs = zeros(T, porder+1, 2, length(pts))
    
    # Use views to avoid unnecessary copying
    @views nfs[:,1,:] = nf  # Index 1 in Julia (was index 0 in Python)
    @views nfs[:,2,:] = nfx # Index 2 in Julia (was index 1 in Python)
    
    return nfs
end

"""     
shape2d calculates the nodal shapefunctions and its derivatives for 
        the master triangle [0,0]-[1,0]-[0,1]
nfs=shape2d(porder,plocal,pts)

porder:    polynomial order
plocal:    node positions (np,2) (np=(porder+1)*(porder+2)/2)
pts:       coordinates of the points where the shape fucntions
             and derivatives are to be evaluated (npoints,2)
nfs:       shape function adn derivatives (np,3,npoints)
             nsf[:,0,:] shape functions 
             nsf[:,1,:] shape fucntions derivatives w.r.t. x
             nsf[:,2,:] shape fucntions derivatives w.r.t. y
"""
function shape2d(porder, plocal, pts)
    # Calculate number of nodes for this polynomial order
    np = (porder + 1) * (porder + 2) ÷ 2
    
    # Number of evaluation points
    npoints = size(pts, 1)
    
    # Calculate coefficient matrix A for transforming from modal to nodal basis
    # Using Koornwinder polynomials as the modal basis
    W, _, _ = koornwinder2d(plocal[:, 2:3], porder)
    
    # Invert the Vandermonde matrix to get transformation coefficients
    A = W \ I

    # Initialize output array for shape functions and derivatives
    nfs = zeros(np, 3, npoints)
    
    # Evaluate Koornwinder polynomials and their derivatives at requested points
    Λ, Λξ, Λη = koornwinder2d(pts, porder)

    # Transform from modal to nodal basis
    ϕ = (Λ * A)'    # Shape functions
    ϕξ = (Λξ * A)'  # x-derivatives of shape functions
    ϕη = (Λη * A)'  # y-derivatives of shape functions

    # Store results in the output array
    nfs[:, 1, :] .= ϕ
    nfs[:, 2, :] .= ϕξ
    nfs[:, 3, :] .= ϕη
    
    return nfs
end

"""
    get_local_face_nodes(mesh, master, face_number, flip_face_direction=false)

Local (element) indices of the `porder+1` nodes lying on global face
`face_number` of the element to its left (`mesh.f[face_number, 3]`), in
counterclockwise face order — or reversed with `flip_face_direction=true`,
which matches how the right element sees the same face (`master.perm[..., 2]`).
"""
function get_local_face_nodes(mesh, master, face_number, flip_face_direction=false)
    # Get the triangle index that contains this face
    it = mesh.f[face_number, 3]
    
    # Find which local face of the triangle corresponds to the global face number
    # (triangles have 3 faces, each with a local number 1, 2, or 3)
    local_face_number = findfirst(x -> x == face_number, mesh.t2f[it, :])
    
    # Get the nodes on this face in the appropriate order
    # flip_face_direction=true uses reversed ordering, important for maintaining
    # consistent orientation between elements sharing a face
    if flip_face_direction
        node_numbers = master.perm[:, local_face_number, 2]  # Reversed ordering
    else
        node_numbers = master.perm[:, local_face_number, 1]  # Standard ordering
    end

    return node_numbers
end

function rotate_plocal(plocal::AbstractArray{T,2}, porder::Integer) where T <: AbstractFloat
    # Create matrix m filled with -1
    m = fill(-1, porder+1, porder+1)
    
    # Fill m with indices (0-based as in Python)
    k = 0
    for i in porder:-1:0
        for j in 0:porder
            if j <= i
                m[i+1, j+1] = k
                k += 1
            end
        end
    end
    
    # Create rotated version of m (equivalent to np.flipud(rm.T))
    rm = reverse(transpose(copy(m)), dims=1)
    
    # Roll each row of rm
    for i in porder-1:-1:0
        rm[i+1, :] = circshift(rm[i+1, :], i+1)
    end
    
    # Create new point array
    plocal_n = zeros(T, size(plocal))
    
    # Rearrange points
    for i in 0:porder
        for j in 0:porder
            idx = m[i+1, j+1]
            if idx > -1
                rm_idx = rm[i+1, j+1]
                plocal_n[idx+1, :] = circshift(plocal[rm_idx+1, :], 1)
            end
        end
    end
    
    return plocal_n
end

"""
Compute node positions on the master volume element.

# Arguments
- `porder::Integer`: Polynomial order
- `nodetype::Integer=0`: Flag determining node distribution:
     - nodetype = 0: Uniform distribution (default)
     - nodetype = 1: Extended Chebyshev nodes of the first kind
     - nodetype = 2: Extended Chebyshev nodes of the second kind

# Returns
- `plocal::Vector{Float64}`: Vector of node positions on the master volume element
"""
function localpnts1d(porder::Integer, nodetype::Integer=0)::Vector{Float64}
    if nodetype == 0
        # Uniform distribution
        plocal = collect(LinRange(0, 1, porder+1))
    elseif nodetype == 1
        # Extended Chebyshev nodes of the first kind
        k = Float64.([i for i in 0:porder])
        denominator = cos(π / (2.0 * (porder+1)))
        plocal = -cos.((2.0 .* k .+ 1.0) .* π ./ (2.0 * (porder+1))) ./ denominator
        plocal = 0.5 .+ 0.5 .* plocal
    elseif nodetype == 2
        # Extended Chebyshev nodes of the second kind
        k = Float64.([porder-i for i in 0:porder])
        plocal = cos.(π .* k ./ porder)
        plocal = 0.5 .+ 0.5 .* plocal
    else
        error("Invalid node type. Valid options are 0, 1, or 2.")
    end
    
    return plocal
end

"""
localpnts 2-d mesh generator for the master element.
Returns (plocal, tlocal) where:
  plocal:    node positions (npl,3) in barycentric coordinates
  tlocal:    triangle indices (nt,3)
  porder:    order of the complete polynomial
             npl = (porder+1)*(porder+2)/2
"""
function localpnts(porder::Integer, nodetype::Integer=0)

    # Get 1D points
    ploc1d = localpnts1d(porder, nodetype)
    
    # Create mesh grid (equivalent to np.meshgrid)
    u = [x for y in ploc1d, x in ploc1d]'  # Each row is ploc1d
    v = [y for y in ploc1d, x in ploc1d]'  # Each column is ploc1d
    uf = vec(u)
    vf = vec(v)
    
    # Create barycentric coordinates [1-u-v, u, v]
    plocal = hcat(1 .- uf .- vf, uf, vf)
    
    # Filter valid points (where first coordinate > -1.0e-10)
    ind = findall(x -> x > -1.0e-10, plocal[:,1])
    plocal = plocal[ind, :]
    
    # Rotate and average to preserve symmetry
    plocal_1 = rotate_plocal(plocal, porder)
    plocal_2 = rotate_plocal(plocal_1, porder)
    plocal = (plocal + plocal_1 + plocal_2) / 3.0

    plocal[abs.(plocal) .< eps(typeof(plocal[1]))] .= 0.0  # Set very small values to zero
    
    # Create triangulation
    shf = 0
    tlocal = zeros(Int, 0, 3)
    
    for jj in 0:porder-1
        ii = porder - jj
        
        # First set of triangles
        l1 = zeros(Int, ii, 3)
        for i in 0:ii-1
            l1[i+1, :] = [i, i+1, ii+i+1] .+ shf
        end
        tlocal = vcat(tlocal, l1)
        
        # Second set of triangles (if applicable)
        if ii > 1
            l2 = zeros(Int, ii-1, 3)
            for i in 0:ii-2
                l2[i+1, :] = [i+1, ii+i+2, ii+i+1] .+ shf
            end
            tlocal = vcat(tlocal, l2)
        end
        
        shf += ii + 1
    end

    tlocal .+= 1  # Convert to 1-based indexing
    
    return plocal, tlocal
end