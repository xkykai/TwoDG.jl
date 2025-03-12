using LinearAlgebra

function unique_rows(A::AbstractMatrix{T}; return_index::Bool=false, return_inverse::Bool=false) where T
    # Input validation
    ndims(A) == 2 || throw(ArgumentError("array must be 2-dimensional"))
    
    # If no indices are requested, use Julia's built-in unique function
    if !return_index && !return_inverse
        return unique(A, dims=1)  # Julia's unique already sorts
    end
    
    # Otherwise, implement custom logic
    m, n = size(A)
    
    # Convert rows to tuples for easier handling and uniqueness checking
    row_tuples = Vector{NTuple{n,T}}(undef, m)
    @inbounds for i in 1:m
        row_tuples[i] = Tuple(@view A[i,:])
    end
    
    # Find unique tuples and their first occurrence indices
    seen = Dict{NTuple{n,T}, Int}()
    I = Int[]  # Indices of first occurrences
    unique_tuples = NTuple{n,T}[]
    
    @inbounds for i in 1:m
        tuple = row_tuples[i]
        if !haskey(seen, tuple)
            push!(unique_tuples, tuple)
            push!(I, i)
            seen[tuple] = length(unique_tuples)
        end
    end
    
    # Sort the unique tuples lexicographically
    p = sortperm(unique_tuples)
    
    # Apply the permutation
    sorted_unique_tuples = unique_tuples[p]
    sorted_I = I[p]
    
    # Create output matrix of sorted unique rows
    B = Matrix{T}(undef, length(sorted_unique_tuples), n)
    @inbounds for i in 1:length(sorted_unique_tuples)
        tuple = sorted_unique_tuples[i]
        for j in 1:n
            B[i,j] = tuple[j]
        end
    end
    
    # Compute inverse mapping if requested
    if return_inverse
        # Create a mapping from original tuples to their sorted index
        tuple_to_idx = Dict(tuple => i for (i, tuple) in enumerate(sorted_unique_tuples))
        J = Vector{Int}(undef, m)
        @inbounds for i in 1:m
            J[i] = tuple_to_idx[row_tuples[i]]
        end
    end
    
    # Return requested combinations
    if return_index
        if return_inverse
            return B, sorted_I, J
        else
            return B, sorted_I
        end
    else
        if return_inverse
            return B, J
        else
            return B
        end
    end
end