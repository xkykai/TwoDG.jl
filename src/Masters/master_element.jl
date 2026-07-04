using LinearAlgebra
using TwoDG.Meshes: Mesh

"""
    ReferenceElement(porder; pgauss=max(4porder, 1), nodetype=0)
    ReferenceElement(plocal, porder; pgauss=max(4porder, 1))
    ReferenceElement(mesh::Mesh, pgauss=nothing)

Reference (master) triangle of polynomial order `porder`: tabulated shape
functions, quadrature rules, and the local node orderings the solvers need.
The element is mesh-independent; the `Mesh` convenience constructor just reads
`mesh.porder`/`mesh.plocal` so the element's nodes match the mesh's `dgnodes`.

`pgauss` is the polynomial degree integrated exactly by the quadrature rules;
`nodetype` selects the node distribution (`0` uniform, `1`/`2` extended
Chebyshev, see [`localpnts`](@ref)); `plocal` may be passed directly to use a
custom node set.

# Fields and conventions
- `porder` — polynomial order; `npl = (porder+1)(porder+2)/2` nodes.
- `plocal :: (npl, 3)` — node positions in barycentric coordinates.
- `corner :: NTuple{3, Int}` — indices of the three vertex nodes in `plocal`.
- `perm :: (porder+1, 3, 2)` — face-node orderings: `perm[:, j, 1]` lists the
  nodes on local face `j` traversed counterclockwise, `perm[:, j, 2]` the
  same nodes reversed (used when a neighboring element sees the shared face
  with opposite orientation — the sign of `mesh.t2f`).
- `ploc1d :: (porder+1, 2)` — 1D node positions on a face.
- `gp1d`, `gw1d` — 1D (face) quadrature points and weights on ``[0, 1]``.
- `gpts`, `gwgh` — 2D (volume) quadrature points and weights on the
  reference triangle.
- `sh1d :: (porder+1, 2, ng1d)` — 1D shape functions (`[:, 1, :]`) and their
  derivatives (`[:, 2, :]`) at the face quadrature points.
- `shap :: (npl, 3, ng2d)` — 2D shape functions (`[:, 1, :]`) and their
  ``ξ``/``η`` derivatives (`[:, 2, :]` / `[:, 3, :]`) at the volume
  quadrature points.
- `mass :: (npl, npl)` — reference-element mass matrix.
- `conv :: (npl, npl, 2)` — reference convection matrices
  (``∫ φᵢ ∂φⱼ/∂ξ`` and ``∫ φᵢ ∂φⱼ/∂η``).
- `ma1d :: (porder+1, porder+1)` — 1D face mass matrix.
"""
struct ReferenceElement{T <: AbstractFloat}
    porder :: Int
    plocal :: Matrix{T}
    corner :: NTuple{3, Int}
      perm :: Array{Int, 3}
    ploc1d :: Matrix{T}
      gp1d :: Vector{T}
      gpts :: Matrix{T}
      gw1d :: Vector{T}
      gwgh :: Vector{T}
      sh1d :: Array{T, 3}
      shap :: Array{T, 3}
      mass :: Matrix{T}
      conv :: Array{T, 3}
      ma1d :: Matrix{T}
end

function ReferenceElement(plocal::AbstractMatrix{<:Real}, porder::Integer;
                          pgauss::Integer=max(4porder, 1))
    T = float(eltype(plocal))
    plocal = Matrix{T}(plocal)
    npl = size(plocal, 1)
    npl == (porder + 1) * (porder + 2) ÷ 2 ||
        throw(ArgumentError("plocal has $npl nodes; order $porder needs $((porder + 1) * (porder + 2) ÷ 2)"))

    # vertex nodes: the ones with a barycentric coordinate equal to 1
    corner = ntuple(3) do i
        c = findfirst(>(1 - 1e-6), @view plocal[:, i])
        c === nothing && throw(ArgumentError("plocal has no vertex node for corner $i"))
        c
    end

    # face j is the edge where barycentric coordinate j vanishes; traverse it
    # counterclockwise (sheet 1), and reversed (sheet 2) for the neighboring
    # element that sees the shared face with opposite orientation
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
    ma1d = sh1d[:, 1, :] * Diagonal(gw1d) * sh1d[:, 1, :]'

    return ReferenceElement{T}(Int(porder), plocal, corner, perm, ploc1d,
                               Vector{T}(gp1d), Matrix{T}(gpts), Vector{T}(gw1d),
                               Vector{T}(gwgh), sh1d, shap, mass, conv, ma1d)
end

ReferenceElement(porder::Integer; nodetype::Integer=0, pgauss::Integer=max(4porder, 1)) =
    ReferenceElement(first(localpnts(porder, nodetype)), porder; pgauss)

ReferenceElement(mesh::Mesh, pgauss=nothing) =
    ReferenceElement(mesh.plocal, mesh.porder;
                     pgauss=pgauss === nothing ? max(4 * mesh.porder, 1) : pgauss)

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