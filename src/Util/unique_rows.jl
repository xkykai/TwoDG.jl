using LinearAlgebra

function unique_rows(A::AbstractMatrix{T}; return_index::Bool=false, return_inverse::Bool=false) where T
    # Input validation
    ndims(A) == 2 || throw(ArgumentError("array must be 2-dimensional"))

    # Use Set operations for faster unique checking
    if return_index || return_inverse
        # Preallocate for better performance
        m, n = size(A)
        
        # Create a Vector of tuples using views for memory efficiency
        # Using @views to avoid allocations when slicing
        rows = Vector{NTuple{n,T}}(undef, m)
        @inbounds for i in 1:m
            rows[i] = Tuple(@view A[i,:])
        end
        # Use Dict for O(1) lookup
        seen = Dict{NTuple{n,T}, Int}()
        I = Int[]
        J = Vector{Int}(undef, m)
        unique_count = 0
        
        # Single pass through the data
        @inbounds for i in 1:m
            row = rows[i]
            idx = get(seen, row, 0)
            if idx == 0
                unique_count += 1
                seen[row] = unique_count
                push!(I, i)
                J[i] = unique_count
            else
                J[i] = idx
            end
        end
        
        # Create output matrix efficiently
        B = Matrix{T}(undef, length(I), n)
        @inbounds for (i, idx) in enumerate(I)
            for j in 1:n
                B[i,j] = A[idx,j]
            end
        end
        
        # Return requested combinations
        if return_index
            if return_inverse
                return B, I, J
            else
                return B, I
            end
        else
            return B, J
        end
    else
        return unique(A, dims=1)
    end
end