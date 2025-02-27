using Jacobi
using Polynomials
using DSP

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
function koornwinder1d(x, p::Integer)
    # Transform x from [0,1] to [-1,1]
    x_transformed = 2x .- 1
    
    # Preallocate output arrays
    npoints = length(x)
    f = zeros(npoints, p + 1)
    fx = zeros(npoints, p + 1)
    
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
function koornwinder2d(x::AbstractMatrix{<:Real}, p::Int)
    # Transform coordinates from [0,1] to [-1,1]
    x = 2 .* x .- 1.0
    npoints = size(x, 1)
    npol = div((p + 1) * (p + 2), 2)
    
    # Preallocate matrices for function values and derivatives
    f  = zeros(Float64, npoints, npol)
    fx = zeros(Float64, npoints, npol)
    fy = zeros(Float64, npoints, npol)
    
    # Get polynomial order pairs (assumed to be provided by pascalindex)
    pq = pascalindex(npol)
    
    # Copy x to avoid modifying the original array
    xc = copy(x)
    # Adjust second coordinate (column 2 in Julia) to avoid singularities
    xc[:, 2] .= min.(0.99999999, xc[:, 2])
    
    # Set up the evaluation coordinates e.
    # In Python, e[:,0] corresponds to the first column; in Julia, we use column 1.
    e = zeros(Float64, size(xc))
    e[:, 1] .= 2 .* (1.0 .+ xc[:, 1]) ./ (1 .- xc[:, 2]) .- 1.0
    e[:, 2] .= xc[:, 2]
    
    # For points where the original x's second coordinate equals 1,
    # set e accordingly (Python: e[ii,0]=-1, e[ii,1]=1; Julia: columns 1 and 2).
    idx = findall(x[:, 2] .== 1.0)
    for i in idx
        e[i, 1] = -1.0
        e[i, 2] = 1.0
    end
    
    # Build the Vandermonde matrix for the Koornwinder polynomials
    for i in 1:npol
        p_order = pq[i, 1]  # corresponds to pq[ii,0] in Python
        q_order = pq[i, 2]  # corresponds to pq[ii,1] in Python
        # Obtain Jacobi polynomial coefficients for p and q parts.
        # pp = Polynomial(jacobi(p_order, 0, 0))
        # qp = Polynomial(jacobi(q_order, 2 * p_order + 1, 0))
        pp = poly_jacobi(p_order, 0, 0)
        qp = poly_jacobi(q_order, 2 * p_order + 1, 0)
        
        # Convolve qp with [-0.5, 0.5] p_order times.
        # Note Polynomual convention is different between Python and Julia.
        # In Python, the coefficients are in decreasing order, while in Julia, they are in increasing order.
        for j in 1:p_order
            qp = Polynomial(reverse(conv([0.5, -0.5], reverse(qp.coeffs))))
        end
        
        # Evaluate the polynomials at the mapped coordinates.
        pval = pp.(e[:, 1])
        qval = qp.(e[:, 2])
        
        # Compute the scaling factor
        fc = sqrt((2.0 * p_order + 1.0) * 2.0 * (p_order + q_order + 1.0))
        
        f[:, i] .= fc .* pval .* qval
    end

    
    # Compute the derivatives of the mapping from (x₁,x₂) to (e₁,e₂)
    de1 = zeros(size(xc))
    de1[:, 1] .= 2.0 ./ (1 .- xc[:, 2])
    de1[:, 2] .= 2.0 .* (1.0 .+ xc[:, 1]) ./ ((1 .- xc[:, 2]).^2)
    
    # Build the Vandermonde matrices for the derivatives.
    for i in 1:npol
        p_order = pq[i, 1]
        q_order = pq[i, 2]
        
        pp = poly_jacobi(p_order, 0, 0)
        qp = poly_jacobi(q_order, 2 * p_order + 1, 0)
        for j in 1:p_order
            qp = Polynomial(reverse(conv([-0.5, 0.5], reverse(qp.coeffs))))
        end
        # @info "i = $i, qp = $qp"

        # @info "i = $i, pp = $pp"
        # @info "i = $i, qp = $qp"
        
        # Compute derivative polynomials.
        dpp = derivative(pp)
        dqp = derivative(qp)
        
        # @info "i = $i, dpp = $dpp"
        # @info "i = $i, dqp = $dqp"

        # Evaluate polynomials and their derivatives.
        pval  = pp.(e[:, 1])
        qval  = qp.(e[:, 2])
        dpval = dpp.(e[:, 1])
        dqval = dqp.(e[:, 2])

        # @info "i = $i, pval = $(pval[1])"
        # @info "i = $i, qp = $qp, e[:, 2] = $(e[:, 2])"
        # @info "i = $i, qval = $(qval[1])"
        # @info "i = $i, dpval = $(dpval[1])"
        # @info "i = $i, dqval = $(dqval[1])"
        
        fc = sqrt((2.0 * p_order + 1.0) * 2.0 * (p_order + q_order + 1.0))
        # @info "i = $i, fc = $fc"
        # @info "i = $i, de1[:, 1] =  $(de1[:, 1])"
        # @info "i = $i, de1[:, 2] =  $(de1[:, 2])"
        
        # @info "i = $i, fc = $fc, dpval = $dpval, qval = $qval, de1[:, 2] = $(de1[:, 2]), pval = $pval, dqval = $dqval"
        # @info "i = $i, fx[:, i] = $(fx[:, i])"
        # @info "i = $i, fc = $fc, dpval = $dpval, qval = $qval, de1[:, 1] = $(de1[:, 1])"
        fx[:, i] .= fc .* dpval .* qval .* de1[:, 1]
        fy[:, i] .= fc .* (dpval .* qval .* de1[:, 2] .+ pval .* dqval)
    end
    
    # Adjust derivatives by a factor of 2.
    fx .*= 2.0
    fy .*= 2.0
    
    return f, fx, fy
end