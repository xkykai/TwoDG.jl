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
    ind1 = 1:npf  # Julia uses 1-based indexing
    ind2 = npf:-1:1  # Reverse range from npf down to 1
    
    # Reshape element vectors and matrices
    # Julia is column-major by default, matching Python's order='F'
    FE_reshaped = reshape(FE, (nch, npf, nfe, ne))
    AE_reshaped = reshape(AE, (nch, npf, nfe, nch, npf, nfe, ne))
    
    # Pre-allocate global matrix and vector
    A = zeros(ncf, ncf, nbf, nf)
    F = zeros(ncf, nf)
    
    # Parallel implementation
    @views Threads.@threads for i in 1:nf
        A_i, F_i = global_assembly(AE_reshaped, FE_reshaped, f, t2f, ind1, ind2, ncf, nbf, nfe, i)
        A[:, :, :, i] .= A_i
        F[:, i] .= F_i
    end

    return A, vec(F)
end

"""
    hdg_parsolve(master, mesh, source, dbc, param)

Solves the convection-diffusion equation using the HDG method.

# Arguments
- `master`: master structure
- `mesh`: mesh structure
- `source`: source term
- `dbc`: dirichlet data
- `param`: dictionary with parameters:
  - `param[:kappa]`: diffusivity coefficient
  - `param[:c]`: convective velocity

# Returns
- `uh`: approximate scalar variable
- `qh`: approximate flux
- `uhath`: approximate trace
"""
@inline function hdg_parsolve(master, mesh, source, dbc, param; kwargs...)
    # Get mesh dimensions
    nps = mesh.porder + 1
    npl = size(mesh.dgnodes, 1)
    nt = size(mesh.t, 1)
    nf = size(mesh.f, 1)

    # Pre-allocate element matrices and vectors
    ae = zeros(3*nps, 3*nps, nt)
    fe = zeros(3*nps, nt)

    # Get number of available threads
    num_threads = Threads.nthreads()
    println("Number of parallel workers: $num_threads")
    
    # Element matrix computation in parallel
    @views Threads.@threads for i in 1:nt
        ae_i, fe_i = elemmat_hdg(view(mesh.dgnodes, :, :, i), master, source, param)
        ae[:, :, i] = ae_i
        fe[:, i] = fe_i
    end

    # Apply boundary conditions
    # Find first boundary face
    ni = findfirst(f -> f[4] < 0, eachrow(mesh.f))
    
    @views Threads.@threads for i in ni:size(mesh.f, 1)
        el = mesh.f[i, 3]  # Element index (adjusted for 1-based indexing)
        ipl = sum(mesh.t[el, :]) - sum(mesh.f[i, 1:2])
        isl = findfirst(x -> x == ipl, mesh.t[el, :])
        
        # Get the nodes on this boundary face
        face_nodes = master.perm[:, isl, 1]  # Get local indices of nodes on this face
        
        # Extract physical coordinates of the face nodes
        face_coords = mesh.dgnodes[face_nodes, :, el]
        
        # Evaluate the Dirichlet boundary condition at these coordinates
        bc_values = dbc(face_coords)
        
        # Clear row and set identity block on diagonal
        ae[(isl-1)*nps+1:isl*nps, :, el] .= 0
        ae[(isl-1)*nps+1:isl*nps, (isl-1)*nps+1:isl*nps, el] = I(nps)
        
        # Set the right-hand side to the boundary condition values
        fe[(isl-1)*nps+1:isl*nps, el] = bc_values
    end

    # Solve global system
    uhath, _ = hdg_gmres(ae, fe, mesh.t2f, mesh.f, nps, f2f=mesh.f2f; kwargs...)

    # Connectivity array for trace variable
    elcon = zeros(Int, 3*nps, nt)

    # Process t2f entries
    Threads.@threads for i in 1:nt
        for j in 1:3
            f = mesh.t2f[i, j]
            if f > 0
                elcon[(j-1)*nps+1:j*nps, i] .= (f-1)*nps+1:f*nps
            elseif f < 0
                f = abs(f)  # Get positive face index
                elcon[(j-1)*nps+1:j*nps, i] .= f*nps:-1:(f-1)*nps+1
            end
        end
    end

    # Solve local problems to get uh and qh
    uh = zeros(npl, nt)
    qh = zeros(npl, 2, nt)
  
    # Local problem computation in parallel
    @views Threads.@threads for i in 1:nt
        uhath_local = uhath[elcon[:, i]]
        uh_i, qh_i = localprob(mesh.dgnodes[:, :, i], master, uhath_local, source, param)
        uh[:, i] .= uh_i
        qh[:, :, i] .= qh_i
    end

    return uh, qh, uhath
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
    
    # Reshape F from flattened vector to 2D array
    F_2d = reshape(F, ncf, nf)
    
    # Initialize result vector in 2D form
    v_2d = zeros(eltype(F), ncf, nf)
    
    # Parallel loop over all faces
    @views Threads.@threads for i in 1:nf
        local_result = zeros(eltype(v_2d), ncf)
        for k in 1:size(f2f, 2)
            j = f2f[i, k]
            if j > 0
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
    
    # Reshape without allocation
    F_2d = reshape(F, ncf, nf)
    result_2d = reshape(result, ncf, nf)
    result_2d .= 0
    
    # Thread-local buffers
    Threads.@threads for i in 1:nf
        local_result = zeros(eltype(result_2d), ncf)
        
        for k in 1:size(f2f, 2)
            j = f2f[i, k]
            if j > 0
                mul!(local_result, view(A, :, :, k, i), view(F_2d, :, j), 1.0, 1.0)
            end
        end
        
        result_2d[:, i] .= local_result
    end
end

"""
    hdg_gmres(AE, FE, t2f, f, npf; 
              x=nothing, restart=160, tol=1e-6, maxit=1000)

HDG GMRES solver with block Jacobi preconditioning.

# Arguments
- `AE::Array`: Element matrices
- `FE::Array`: Element vectors
- `t2f::Array{Int}`: Element to face connectivity
- `f::Array{Int}`: Face to element connectivity
- `npf::Int`: Number of points per face
- `x::Union{Array,Nothing}=nothing`: Initial guess (optional)
- `restart::Int=160`: Restart parameter
- `tol::Float64=1e-6`: Tolerance
- `maxit::Int=1000`: Maximum iterations

# Returns
- `x::Array`: Solution vector
- `flags::Int`: Flag indicating convergence status
- `iter::Int`: Number of iterations
- `rev::Array`: Residual history
"""
@inline function hdg_gmres(AE, FE, t2f, f, npf; x=nothing, restart=80, tol=1e-6, maxit=2000, f2f=nothing)
    # Assemble the global system in dense format
    A, b = hdg_densesystem(AE, FE, f, t2f, npf)

    # Compute the block Jacobi preconditioner
    B = compute_blockjacobi(A)
    
    # Make face-to-face connectivities
    if f2f === nothing
        f2f = mkf2f(f, t2f)
    end

    N = length(b)
    if isnothing(x)
        x = zeros(N)
    end
    
    ortho = 1 # 1 -> MGS orthogonalization, otherwise CGS orthogonalization
    b0 = copy(b)
    
    # Apply preconditioner to b
    apply_blockjacobi!(b, B, b)
    nrmb = norm(b)
    
    # Pre-allocate ALL temporary arrays
    H = zeros(restart+1, restart)
    v = zeros(N, restart+1)
    e1 = zeros(restart+1)
    e1[1] = 1.0
    rev = zeros(restart)
    cs = ones(restart+1)
    sn = zeros(restart+1)
    H_col = zeros(restart+1)
    
    # Pre-allocate for matrix-vector product
    d = zeros(N)
    r = zeros(N)
    
    # Pre-allocate for solution update
    y_full = zeros(restart)
    
    flags = 10
    iter_count = 0
    cycle = 0
    
    @views while true
        # Reuse pre-allocated arrays for matrix-vector product
        hdg_matvec!(d, A, x, f2f)
        
        # Apply preconditioner in-place
        apply_blockjacobi!(d, B, d)

        # Compute residual in-place
        r .= b .- d
        
        beta = norm(r)
        v[:, 1] .= r ./ beta
        res = beta
        iter_count += 1
        
        if iter_count <= length(rev)
            rev[iter_count] = res
        end
        
        g = beta .* e1
        y_length = 0

        for j in 1:restart
            # Reuse pre-allocated array for matrix-vector product
            hdg_matvec!(d, A, view(v, :, j), f2f)
            
            # Apply preconditioner in-place
            apply_blockjacobi!(d, B, d)
            
            v[:, j+1] .= d
            
            # Arnoldi process using BLAS operations for better performance
            arnoldi!(H, v, j, ortho)
            H[j+1, j] = norm(view(v, :, j+1))

            if H[j+1, j] != 0.0
                v[:, j+1] ./= H[j+1, j]
            else
                break
            end

            # Use pre-allocated H_col
            H_col[1:j+1] .= view(H, 1:(j+1), j)
            
            # Apply Givens rotations in-place
            givens_rotation!(H_col[1:j+1], g, cs, sn, j)
            H[1:(j+1), j] .= H_col[1:j+1]

            y_length = j

            # Check convergence with reused variables
            res = abs(g[j+1])
            iter_count += 1
            
            if iter_count <= length(rev)
                rev[iter_count] = res
            end
            
            # Check convergence
            if res / nrmb <= tol
                flags = 0
                break
            end
            
            # Check maximum iterations
            if iter_count >= maxit
                flags = 1
                break
            end
        end
        
        # Use pre-allocated y_full for solution
        y_view = view(y_full, 1:y_length)
        back_solve!(y_view, view(H, 1:y_length, 1:y_length), view(g, 1:y_length))
        
        # Update solution in-place
        x .+= v[:, 1:y_length] * y_view
        cycle += 1
        
        if flags < 10
            println("gmres($restart) converges at $iter_count iterations with relative residual $(res/nrmb)")
            break
        end
    end
    
    # Compute final residual
    rev ./= nrmb
    hdg_matvec!(d, A, x, f2f)
    r = vec(b0) - vec(d)
    final_residual = norm(r)
    println("Final residual: $final_residual")

    return x, flags, iter_count, rev
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
        # Use factorization instead of direct inversion - much more efficient
        A_i = view(A, :, :, 1, i)
        
        try
            # Use LU factorization and solve against identity
            F = lu(A_i)
            for j in 1:ncf
                # Solve one column at a time for better cache locality
                col_view = view(B, :, j, i)
                col_view .= 0
                col_view[j] = 1.0
                ldiv!(F, col_view)
            end
        catch
            # Handle singular matrices by adding regularization
            tmp_mat .= A_i
            for j in 1:ncf
                tmp_mat[j,j] += 1e-12
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

"""
    arnoldi(H, v, j, ortho=1)

Performs the Arnoldi process to orthogonalize v[:, j+1] against previous basis vectors.

# Arguments
- `H::AbstractMatrix`: Hessenberg matrix
- `v::AbstractMatrix`: Krylov subspace basis vectors
- `j::Integer`: Current iteration
- `ortho::Integer=1`: Orthogonalization method (1: MGS, 0: CGS)

# Returns
- `H::AbstractMatrix`: Updated Hessenberg matrix
- `v::AbstractMatrix`: Updated basis vectors
"""
@inline function arnoldi(H::AbstractMatrix, v::AbstractMatrix, j::Integer, ortho::Integer=1)
    if ortho == 1
        # Modified Gram-Schmidt (MGS)
        # Sequentially orthogonalize against each previous vector
        for i in 1:j
            H[i, j] = dot(v[:, j+1], v[:, i])
            v[:, j+1] .= v[:, j+1] .- H[i, j] .* v[:, i]
        end
    else
        # Classical Gram-Schmidt (CGS)
        # Compute all projections at once
        H[1:j, j] = v[:, 1:j]' * v[:, j+1]
        # Orthogonalize against all previous vectors at once
        v[:, j+1] = v[:, j+1] - v[:, 1:j] * H[1:j, j]
    end

    return H, v
end

"""
    arnoldi!(H, v, j, ortho=1)

Performs the Arnoldi process to orthogonalize v[:, j+1] against previous basis vectors in a mutating way.

# Arguments
- `H::AbstractMatrix`: Hessenberg matrix
- `v::AbstractMatrix`: Krylov subspace basis vectors
- `j::Integer`: Current iteration
- `ortho::Integer=1`: Orthogonalization method (1: MGS, 0: CGS)
"""
@inline function arnoldi!(H::AbstractMatrix, v::AbstractMatrix, j::Integer, ortho::Integer=1)
    @views if ortho == 1
        # Modified Gram-Schmidt (MGS)
        # Sequentially orthogonalize against each previous vector
        for i in 1:j
            H[i, j] = v[:, j+1] ⋅ v[:, i]
            v[:, j+1] .= v[:, j+1] .- H[i, j] .* v[:, i]
        end
    else
        # Classical Gram-Schmidt (CGS)
        # Compute all projections at once
        H[1:j, j] = v[:, 1:j]' * v[:, j+1]
        # Orthogonalize against all previous vectors at once
        v[:, j+1] = v[:, j+1] - v[:, 1:j] * H[1:j, j]
    end
end

"""
    givens_rotation(H, s, cs, sn, i)

Apply Givens rotation to the i-th column of H and update cs, sn, and s.
This is a key component of the GMRES algorithm that transforms the Hessenberg
matrix to upper triangular form via a series of plane rotations.

# Arguments
- `H`: Current column of the Hessenberg matrix
- `s`: Residual vector
- `cs`: Cosine values for rotations
- `sn`: Sine values for rotations
- `i`: Current iteration index

# Returns
- `H`: Updated Hessenberg matrix column
- `s`: Updated residual vector
- `cs`: Updated cosine values
- `sn`: Updated sine values
"""
@inline function givens_rotation(H, s, cs, sn, i)
    rotation_matrix = zeros(2, 2)
    for k in 1:i-1
        rotation_matrix[1, 1] = cs[k]
        rotation_matrix[1, 2] = sn[k]
        rotation_matrix[2, 1] = -sn[k]
        rotation_matrix[2, 2] = cs[k]
        H[k:k+1] .= rotation_matrix * H[k:k+1]
    end

    cs[i] = abs(H[i]) / sqrt(H[i]^2 + H[i+1]^2)
    sn[i] = H[i+1] / H[i] * cs[i]
    H[i] = cs[i] * H[i] + sn[i] * H[i+1]
    H[i+1] = 0.0

    rotation_matrix[1, 1] = cs[i]
    rotation_matrix[1, 2] = sn[i]
    rotation_matrix[2, 1] = -sn[i]
    rotation_matrix[2, 2] = cs[i]

    s[i:i+1] .= rotation_matrix * [s[i], 0]
    return H, s, cs, sn
end

"""
    givens_rotation(H, s, cs, sn, i)

Apply Givens rotation to the i-th column of H and update cs, sn, and s.
This is a key component of the GMRES algorithm that transforms the Hessenberg
matrix to upper triangular form via a series of plane rotations.

# Arguments
- `H`: Current column of the Hessenberg matrix
- `s`: Residual vector
- `cs`: Cosine values for rotations
- `sn`: Sine values for rotations
- `i`: Current iteration index

# Returns
- `H`: Updated Hessenberg matrix column
- `s`: Updated residual vector
- `cs`: Updated cosine values
- `sn`: Updated sine values
"""
@inline function givens_rotation!(H, s, cs, sn, i)
    rotation_matrix = zeros(2, 2)
    @views for k in 1:i-1
        rotation_matrix[1, 1] = cs[k]
        rotation_matrix[1, 2] = sn[k]
        rotation_matrix[2, 1] = -sn[k]
        rotation_matrix[2, 2] = cs[k]
        H[k:k+1] = rotation_matrix * H[k:k+1]
    end

    cs[i] = abs(H[i]) / sqrt(H[i]^2 + H[i+1]^2)
    sn[i] = H[i+1] / H[i] * cs[i]
    H[i] = cs[i] * H[i] + sn[i] * H[i+1]
    H[i+1] = 0.0

    rotation_matrix[1, 1] = cs[i]
    rotation_matrix[1, 2] = sn[i]
    rotation_matrix[2, 1] = -sn[i]
    rotation_matrix[2, 2] = cs[i]

    s[i:i+1] = rotation_matrix * [s[i], 0]
end

"""
    back_solve(H, s)

Solves the upper triangular system Hy = s using back substitution.

# Arguments
- `H`: Upper triangular matrix (Hessenberg matrix after Givens rotations)
- `s`: Right-hand side vector

# Returns
- `y`: Solution vector
"""
@inline function back_solve!(y::AbstractVector, H::AbstractMatrix, s::AbstractVector)
    n = length(s)
    
    # Back substitution in-place
    for i in n:-1:1
        y[i] = s[i]
        
        # Use BLAS dot product for better performance
        for j in i+1:n
            y[i] -= H[i, j] * y[j]
        end
        
        y[i] /= H[i, i]
    end
end