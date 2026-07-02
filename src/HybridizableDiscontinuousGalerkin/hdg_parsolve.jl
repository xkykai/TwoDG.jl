using TwoDG.Meshes: mkf2f
using LinearAlgebra

"""
Assembles the global matrix and vector for a specific face.

Parameters:
-----------
AE : Array
    Element matrices
FE : Array
    Element vectors
f : Array
    Face to element connectivity
t2f : Array
    Element to face connectivity
ind1 : Array
    Forward indices
ind2 : Array
    Backward indices
ncf : Int
    Number of components per face
nbf : Int
    Number of neighboring faces
nfe : Int
    Number of faces per element
i : Int
    Face index

Returns:
--------
A_i : Array
    Global matrix for face i
F_i : Array
    Global vector for face i
"""
@inline function global_assembly(AE, FE, f, t2f, ind1, ind2, ncf, nbf, nfe, i)
    A = zeros(ncf, ncf, nbf)
    
    # Obtain two elements sharing the same face i
    fi = @view f[i, end-1:end]
    
    if fi[2] > 0  # face i is an interior face
        # Obtain neighboring faces
        kf = abs.(t2f[fi, :])
        
        # Obtain the index of face i in elements
        i1 = findfirst(x -> x == i, kf[1, :])
        i2 = findfirst(x -> x == i, kf[2, :])
        
        # Determine orientation
        j1, j2 = t2f[fi[1], i1] > 0 ? (ind1, ind2) : (ind2, ind1)
        
        # First block
        k = 1
        A[:, :, 1] .= reshape(AE[:, j1, i1, :, j1, i1, fi[1]] .+ 
                            AE[:, j2, i2, :, j2, i2, fi[2]], (ncf, ncf))
        F = reshape(FE[:, j1, i1, fi[1]] .+ FE[:, j2, i2, fi[2]], (ncf,))
        
        # Loop over each face of the 1st element
        for is = 1:nfe
            if is != i1
                k += 1
                j3 = t2f[fi[1], is] > 0 ? ind1 : ind2
                A[:, :, k] = reshape(AE[:, j1, i1, :, j3, is, fi[1]], (ncf, ncf))
            end
        end
        
        # Loop over faces of the 2nd element
        for is = 1:nfe
            if is != i2
                k += 1
                j4 = t2f[fi[2], is] > 0 ? ind1 : ind2
                A[:, :, k] = reshape(AE[:, j2, i2, :, j4, is, fi[2]], (ncf, ncf))
            end
        end
    else  # face i is a boundary face
        # Obtain neighboring faces
        kf = abs.(t2f[fi[1], :])
        
        # Obtain the index of face i in the 1st element
        i1 = findfirst(x -> x == i, kf)

        # Determine orientation
        j1 = t2f[fi[1], i1] > 0 ? ind1 : ind2
        
        # First block
        k = 1
        A[:, :, 1] = reshape(AE[:, j1, i1, :, j1, i1, fi[1]], (ncf, ncf))
        F = reshape(FE[:, j1, i1, fi[1]], (ncf,))
        
        # Loop over each face of the 1st element
        for is = 1:nfe
            if is != i1
                k += 1
                j3 = t2f[fi[1], is] > 0 ? ind1 : ind2
                A[:, :, k] = reshape(AE[:, j1, i1, :, j3, is, fi[1]], (ncf, ncf))
            end
        end
    end

    return A, F
end

"""
    hdg_densesystem(AE, FE, f, t2f, npf)

Assembles the global system in dense format.

# Arguments
- `AE`: Element matrices
- `FE`: Element vectors
- `f`: Face to element connectivity
- `t2f`: Element to face connectivity
- `npf`: Number of points per face

# Returns
- `A`: Global matrix in dense format
- `F`: Global vector in dense format
"""
@inline function hdg_densesystem(AE::AbstractArray, FE::AbstractArray, f::AbstractArray, 
                         t2f::AbstractArray, npf::Integer)
    # Get dimensions
    nf = size(f, 1)  # Number of faces
    ne, nfe = size(t2f)  # Number of elements, number of faces per element
    
    # Calculate derived dimensions
    N = length(FE)
    ndf = npf * nfe  # Number of points per face times number of faces per element
    nch = N ÷ (ndf * ne)  # Number of components of UH (integer division)
    ncf = nch * npf  # Number of components of UH times number of points per face
    nbf = 2 * nfe - 1  # Number of neighboring faces
    
    # Create index arrays for face orientation
    ind1 = 1:npf  # Forward indices
    ind2 = npf:-1:1  # Reverse indices for handling differently oriented faces
    
    # Reshape element matrices and vectors to access them by components, points, faces, and elements
    FE_reshaped = reshape(FE, (nch, npf, nfe, ne))
    AE_reshaped = reshape(AE, (nch, npf, nfe, nch, npf, nfe, ne))
    
    # Pre-allocate global matrix and vector
    A = zeros(ncf, ncf, nbf, nf)
    F = zeros(ncf, nf)
    
    # Process each face in parallel, assembling its contribution to the global system
    @views Threads.@threads for i in 1:nf
        A_i, F_i = global_assembly(AE_reshaped, FE_reshaped, f, t2f, ind1, ind2, ncf, nbf, nfe, i)
        A[:, :, :, i] .= A_i
        F[:, i] .= F_i
    end

    return A, vec(F)
end

"""
    hdg_elemmats(master, mesh, source, dbc, param)

Computes the HDG element matrices/vectors `ae (3nps, 3nps, nt)`, `fe (3nps, nt)`
for all elements (threaded) and applies the strong Dirichlet boundary
conditions `dbc` to the boundary-face rows.
"""
function hdg_elemmats(master, mesh, source, dbc, param)
    nps = mesh.porder + 1
    nt = size(mesh.t, 1)

    ae = zeros(3*nps, 3*nps, nt)
    fe = zeros(3*nps, nt)

    # Compute local element matrices in parallel
    @views Threads.@threads for i in 1:nt
        ae_i, fe_i = elemmat_hdg(view(mesh.dgnodes, :, :, i), master, source, param)
        ae[:, :, i] = ae_i
        fe[:, i] = fe_i
    end

    hdg_applydbc!(ae, fe, master, mesh, dbc)

    return ae, fe
end

"""
    hdg_applydbc!(ae, fe, master, mesh, dbc)

Applies the strong Dirichlet boundary condition `dbc` to the boundary-face
rows of the element matrices/vectors `ae`, `fe` in place.
"""
function hdg_applydbc!(ae, fe, master, mesh, dbc)
    nps = mesh.porder + 1

    # Find first boundary face
    ni = findfirst(f -> f[4] < 0, eachrow(mesh.f))

    @views Threads.@threads for i in ni:size(mesh.f, 1)
        el = mesh.f[i, 3]  # Element index
        # Find local face number through point indices
        ipl = sum(mesh.t[el, :]) - sum(mesh.f[i, 1:2])
        isl = findfirst(x -> x == ipl, mesh.t[el, :])

        # Get the nodes on this boundary face
        face_nodes = master.perm[:, isl, 1]  # Local indices of nodes on this face

        # Extract physical coordinates of the face nodes
        face_coords = mesh.dgnodes[face_nodes, :, el]

        # Evaluate the Dirichlet boundary condition at these coordinates
        bc_values = dbc(face_coords)

        # Apply strong Dirichlet BC: clear row and set identity on diagonal
        ae[(isl-1)*nps+1:isl*nps, :, el] .= 0
        ae[(isl-1)*nps+1:isl*nps, (isl-1)*nps+1:isl*nps, el] = I(nps)

        # Set RHS to boundary values
        fe[(isl-1)*nps+1:isl*nps, el] = bc_values
    end

    return ae, fe
end

"""
    hdg_localrecovery(master, mesh, uhath, source, param)

Recovers the element-local solution `uh (npl, 1, nt)` and flux
`qh (npl, 2, nt)` from the global trace vector `uhath` by solving the local
problems (threaded).
"""
function hdg_localrecovery(master, mesh, uhath, source, param)
    nps = mesh.porder + 1
    npl = size(mesh.dgnodes, 1)
    nt = size(mesh.t, 1)

    # Build connectivity array for mapping global trace DOFs to local elements
    elcon = zeros(Int, 3*nps, nt)

    # Process t2f entries to build connectivity
    Threads.@threads for i in 1:nt
        for j in 1:3
            f = mesh.t2f[i, j]
            if f > 0
                # Same orientation - use forward mapping
                elcon[(j-1)*nps+1:j*nps, i] .= (f-1)*nps+1:f*nps
            elseif f < 0
                # Opposite orientation - use reverse mapping
                f = abs(f)  # Get positive face index
                elcon[(j-1)*nps+1:j*nps, i] .= f*nps:-1:(f-1)*nps+1
            end
        end
    end

    # Solve local problems to get uh and qh using the computed trace values
    uh = zeros(npl, 1, nt)
    qh = zeros(npl, 2, nt)

    # Local problem computation in parallel
    @views Threads.@threads for i in 1:nt
        uhath_local = uhath[elcon[:, i]]  # Extract trace values for this element
        uh_i, qh_i = localprob(mesh.dgnodes[:, :, i], master, uhath_local, source, param)
        uh[:, 1, i] .= uh_i
        qh[:, :, i] .= qh_i
    end

    return uh, qh
end

"""
    hdg_parsolve(master, mesh, source, dbc, param;
                 ArrayT=Array, T=Float64, restart=80, tol=1e-6, maxit=2000,
                 preconditioner=true, verbose=false)

Solves the convection-diffusion equation using the HDG method with restarted,
block-Jacobi-preconditioned GMRES (Krylov.jl) on the statically condensed
trace system. The iteration runs on the KernelAbstractions backend of
`ArrayT` (pass `ArrayT=CuArray` with CUDA.jl loaded for a GPU solve); element
assembly and local recovery stay on the CPU.

# Arguments
- `master`: master structure
- `mesh`: mesh structure
- `source`: source term
- `dbc`: dirichlet data
- `param`: dictionary with parameters:
  - `param[:kappa]`: diffusivity coefficient
  - `param[:c]`: convective velocity
  - `param[:taud]`: stabilization parameter

# Returns
- `uh (npl, 1, nt)`: approximate scalar variable
- `qh (npl, 2, nt)`: approximate flux
- `uhath (nps, nf)`: approximate trace
- `niter`: number of GMRES iterations
"""
function hdg_parsolve(master, mesh, source, dbc, param;
                      ArrayT=Array, T::Type{<:AbstractFloat}=Float64, kwargs...)
    nps = mesh.porder + 1

    ae, fe = hdg_elemmats(master, mesh, source, dbc, param)
    sys = adapt(ArrayT, HDGSystem(ae, fe, mesh; T))

    # Solve global system for trace variable (uhath)
    x, stats = hdg_gmres_ka(sys; kwargs...)
    uhath = reshape(Float64.(Array(x)), nps, :)

    uh, qh = hdg_localrecovery(master, mesh, vec(uhath), source, param)

    return uh, qh, uhath, stats.niter
end

"""
    hdg_matvec(A, F, f2f)

Performs matrix-vector multiplication for HDG method using face-to-face connectivity.

# Arguments
- `A`: Global matrix in dense format (ncf, ncf, nbf, nf)
- `F`: Vector to be multiplied (flattened)
- `f2f`: Face-to-face connectivity

# Returns
- `v`: Result of matrix-vector multiplication (flattened)
"""
@inline function hdg_matvec(A, F, f2f)
    nf = size(f2f, 1)   # Number of faces
    ncf = size(A, 1)    # Number of components per face
    
    # Reshape F from flattened vector to 2D array for face-wise operations
    F_2d = reshape(F, ncf, nf)
    
    # Initialize result vector in 2D form
    v_2d = zeros(eltype(F), ncf, nf)
    
    # For each face, compute contribution from neighboring faces
    @views Threads.@threads for i in 1:nf
        local_result = zeros(eltype(v_2d), ncf)
        for k in 1:size(f2f, 2)
            j = f2f[i, k]
            if j > 0  # Skip non-existent neighbors
                # Add contribution from neighboring face j using the k-th block of A
                mul!(local_result, A[:, :, k, i], F_2d[:, j], 1.0, 1.0)
            end
        end
        v_2d[:, i] .= local_result
    end

    # Flatten the result to match expected output format
    return vec(v_2d)
end

@inline function hdg_matvec!(result, A, F, f2f)
    nf = size(f2f, 1)
    ncf = size(A, 1)
    
    # Reshape without allocation - use existing arrays
    F_2d = reshape(F, ncf, nf)
    result_2d = reshape(result, ncf, nf)
    result_2d .= 0
    
    # Thread-local computation to avoid race conditions
    Threads.@threads for i in 1:nf
        local_result = zeros(eltype(result_2d), ncf)
        
        for k in 1:size(f2f, 2)
            j = f2f[i, k]
            if j > 0  # Skip non-existent neighbors
                # Accumulate contributions from each neighbor
                mul!(local_result, view(A, :, :, k, i), view(F_2d, :, j), 1.0, 1.0)
            end
        end
        
        # Copy thread-local result to global result array
        result_2d[:, i] .= local_result
    end
end

"""
    compute_blockjacobi(A)

Computes block Jacobi preconditioner for HDG method.

# Arguments
- `A`: Global matrix in dense format with dimensions (ncf, ncf, nbf, nf)

# Returns
- `B`: Block Jacobi preconditioner with dimensions (ncf, ncf, nf)
"""
@inline function compute_blockjacobi(A)
    ncf = size(A, 1)
    nf = size(A, 4)

    B = zeros(eltype(A), ncf, ncf, nf)
    
    # Pre-allocate thread-local workspace
    tmp_mat = zeros(eltype(A), ncf, ncf)
    
    Threads.@threads for i in 1:nf
        # Get the diagonal block for this face
        A_i = view(A, :, :, 1, i)
        
        try
            # Use LU factorization instead of direct inversion for better numerical stability
            F = lu(A_i)
            for j in 1:ncf
                # Solve against identity columns to effectively compute inverse
                col_view = view(B, :, j, i)
                col_view .= 0
                col_view[j] = 1.0
                ldiv!(F, col_view)
            end
        catch
            # Handle singular or nearly singular matrices with regularization
            tmp_mat .= A_i
            for j in 1:ncf
                tmp_mat[j,j] += 1e-12  # Add small diagonal perturbation
            end
            F = lu(tmp_mat)
            for j in 1:ncf
                col_view = view(B, :, j, i)
                col_view .= 0
                col_view[j] = 1.0
                ldiv!(F, col_view)
            end
        end
    end
    
    return B
end

"""
    apply_blockjacobi(B, v)

Applies a block Jacobi preconditioner to a vector.

# Arguments
- `B::AbstractArray`: Block Jacobi preconditioner with dimensions (ncf, ncf, nf)
- `v::AbstractArray`: Vector to be preconditioned (can be 1D flattened or 2D array)

# Returns
- `w::Array`: Preconditioned vector in the same format as input v
"""
@inline function apply_blockjacobi(B::AbstractArray, v::AbstractArray)
    ncf = size(B, 1)
    nf = size(B, 3)
    
    is_flattened = ndims(v) == 1
    
    # Use reshape to avoid allocation
    v_reshaped = is_flattened ? reshape(v, ncf, nf) : v
    
    # Pre-allocate result with similar type
    w_reshaped = similar(v_reshaped)
    
    # Thread-local computation with minimal allocation
    Threads.@threads for i in 1:nf
        # Direct views to avoid copies
        mul!(view(w_reshaped, :, i), view(B, :, :, i), view(v_reshaped, :, i))
    end
    
    # Return in consistent format without extra allocation
    return is_flattened ? vec(w_reshaped) : w_reshaped
end

@inline function apply_blockjacobi!(result, B::AbstractArray, v::AbstractArray)
    ncf = size(B, 1)
    nf = size(B, 3)
    
    # Reshape without allocation
    v_reshaped = reshape(v, ncf, nf)
    result_reshaped = reshape(result, ncf, nf)
    
    # In-place computation
    Threads.@threads for i in 1:nf
        result_reshaped[:, i] .= B[:, :, i] * v_reshaped[:, i]
    end
end

