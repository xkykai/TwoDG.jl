using Jacobi

"""      
koornwinder1d vandermonde matrix for legenedre polynomials in [0,1]
[f,fx]=koornwinder(x,p)

   x:         coordinates of the points wherethe polynomials 
              are to be evaluated (npoints)
   p:         maximum order of the polynomials consider. that
              is all polynomials of degree up to p, npoly=p+1
   f:         vandermonde matrix (npoints,npoly)
   fx:        vandermonde matrix for the derivative of the koornwinder
              polynomials w.r.t. x (npoints,npoly) 
"""
function koornwinder1d(x::AbstractVector{T}, p::Integer) where T<:Real
    # Transform x from [0,1] to [-1,1]
    x_transformed = 2x .- 1
    
    # Preallocate output arrays
    npoints = length(x)
    f = Matrix{T}(undef, npoints, p + 1)
    fx = Matrix{T}(undef, npoints, p + 1)
    
    # Fill matrices
    for i in 0:p
        # Normalization factor
        norm_factor = sqrt(2i + 1.0)
        
        # Evaluate polynomial and its derivative directly using Jacobi.jl
        f[:, i+1] = jacobi.(x_transformed, i, 0, 0) .* norm_factor
        fx[:, i+1] = djacobi.(x_transformed, i, 0, 0) .* norm_factor
    end
    
    # Scale derivative according to chain rule (dx_transformed/dx = 2)
    fx .*= 2
    
    return f, fx
end

function pascalindex(npol::Integer)
    # Pre-allocate output matrix
    pq = zeros(Int, npol, 2)
    
    # Calculate required polynomial order based on npol
    # Using quadratic formula to solve: n(n+1)/2 ≥ npol
    n = ceil(Int, (-1 + sqrt(1 + 8npol))/2)
    
    idx = 1
    for i in 0:n
        for j in 0:i
            if idx > npol
                return pq
            end
            pq[idx, 1] = i - j
            pq[idx, 2] = j
            idx += 1
        end
    end
    
    return pq
end

"""     
koornwinder2d vandermonde matrix for koornwinder polynomials in 
           the master triangle [0,0]-[1,0]-[0,1]
[f,fx,fy]=koornwinder(x,p)

   x:         coordinates of the points wherethe polynomials 
              are to be evaluated (npoints,dim)
   p:         maximum order of the polynomials consider. that
              is all polynomials of complete degree up to p,
              npoly = (porder+1)*(porder+2)/2
   f:         vandermonde matrix (npoints,npoly)
   fx:        vandermonde matrix for the derivative of the koornwinder
              polynomials w.r.t. x (npoints,npoly)
   fy:        vandermonde matrix for the derivative of the koornwinder
              polynomials w.r.t. y (npoints,npoly)
"""
function koornwinder2d(x::Matrix{T}, p::Integer) where T<:Real
    # Calculate number of polynomials
    npol = div((p + 1) * (p + 2), 2)
    npoints = size(x, 1)
    
    # Transform coordinates and preallocate matrices
    x_transformed = 2x .- 1
    f = Matrix{T}(undef, npoints, npol)
    fx = Matrix{T}(undef, npoints, npol)
    fy = Matrix{T}(undef, npoints, npol)
    
    # Get pascal indices
    pq = pascalindex(npol)
    
    # Handle coordinate transformation
    xc = copy(x_transformed)
    xc[:, 2] = min.(0.99999999, xc[:, 2])  # Displace coordinate for singular node
    
    # Calculate transformed coordinates
    e = similar(xc)
    e[:, 1] = @. 2(1 + xc[:, 1]) / (1 - xc[:, 2]) - 1
    e[:, 2] = xc[:, 2]
    
    # Correct values for singular points
    singular_points = findall(x_transformed[:, 2] .≈ 1.0)
    e[singular_points, 1] .= -1.0
    e[singular_points, 2] .= 1.0
    
    # Calculate function values
    for i in 1:npol
        p_degree = pq[i, 1]
        q_degree = pq[i, 2]
        
        # Calculate normalization factor
        fc = sqrt((2.0 * p_degree + 1.0) * 2.0 * (p_degree + q_degree + 1.0))
        
        # Evaluate Jacobi polynomials directly
        pval = jacobi.(e[:, 1], p_degree, 0, 0)
        qval = jacobi.(e[:, 2], q_degree, 2p_degree + 1, 0)
        
        # Apply shift for q polynomial
        qval .*= (-0.5)^p_degree
        
        # Store results
        f[:, i] = fc .* pval .* qval
    end
    
    # Calculate derivatives
    de1 = similar(xc)
    de1[:, 1] = @. 2.0 / (1 - xc[:, 2])
    de1[:, 2] = @. 2(1 + xc[:, 1]) / (1 - xc[:, 2])^2
    
    for i in 1:npol
        p_degree = pq[i, 1]
        q_degree = pq[i, 2]
        
        # Calculate normalization factor
        fc = sqrt((2.0 * p_degree + 1.0) * 2.0 * (p_degree + q_degree + 1.0))
        
        # Evaluate polynomials and their derivatives
        pval = jacobi.(e[:, 1], p_degree, 0, 0)
        qval = jacobi.(e[:, 2], q_degree, 2p_degree + 1, 0)
        dpval = djacobi.(e[:, 1], p_degree, 0, 0)
        dqval = djacobi.(e[:, 2], q_degree, 2p_degree + 1, 0)
        
        # Apply shift for q polynomial
        qval .*= (-0.5)^p_degree
        dqval .*= (-0.5)^p_degree
        
        # Store derivatives
        fx[:, i] = fc .* (dpval .* qval .* de1[:, 1])
        fy[:, i] = fc .* (dpval .* qval .* de1[:, 2] .+ pval .* dqval)
    end
    
    # Scale derivatives
    fx .*= 2.0
    fy .*= 2.0
    
    return f, fx, fy
end