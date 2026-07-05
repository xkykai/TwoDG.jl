using Jacobi
using Polynomials

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
function pascalindex3d(npol::Integer)
    # triples (p, q, r) of all polynomials of total degree ≤ n, ordered by
    # degree (the 3D analog of pascalindex)
    pqr = zeros(Int, npol, 3)
    idx = 1
    d = 0
    while idx <= npol
        for j in 0:d, k in 0:j
            if idx > npol
                return pqr
            end
            pqr[idx, 1] = d - j
            pqr[idx, 2] = j - k
            pqr[idx, 3] = k
            idx += 1
        end
        d += 1
    end
    return pqr
end

"""
    koornwinder3d(x, p) -> (f, fx, fy, fz)

Vandermonde matrices of the orthonormal PKD (Proriol–Koornwinder–Dubiner)
basis on the unit tetrahedron `[0,0,0]-[1,0,0]-[0,1,0]-[0,0,1]` and of its
three derivatives, at the points `x (npoints, 3)`. All polynomials of
complete degree up to `p`, `npoly = (p+1)(p+2)(p+3)/6`, orthonormal w.r.t.
the unit-tetrahedron measure. Direct extension of [`koornwinder2d`](@ref)'s
collapsed-coordinate + Jacobi-polynomial recipe (Hesthaven & Warburton 2008,
§10.1):

    φ_pqr = c · P_p(a) · ((1-b)/2)^p P_q^{(2p+1,0)}(b) · ((1-c)/2)^{p+q} P_r^{(2p+2q+2,0)}(c)

with `(a, b, c)` the collapsed coordinates of the bi-unit tetrahedron and
`c = sqrt(2(2p+1)(p+q+1)(2(p+q+r)+3))`.
"""
function koornwinder3d(x::AbstractMatrix{<:Real}, p::Int)
    # Transform coordinates from the unit to the bi-unit tetrahedron
    x = 2 .* x .- 1.0
    npoints = size(x, 1)
    npol = div((p + 1) * (p + 2) * (p + 3), 6)

    f  = zeros(Float64, npoints, npol)
    fx = zeros(Float64, npoints, npol)
    fy = zeros(Float64, npoints, npol)
    fz = zeros(Float64, npoints, npol)

    pqr = pascalindex3d(npol)

    # Collapsed coordinates a = -2(1+x₁)/(x₂+x₃) - 1, b = 2(1+x₂)/(1-x₃) - 1,
    # c = x₃. Guard the two collapse singularities (the edge x₂+x₃ = 0 and the
    # vertex x₃ = 1) at roundoff level, as koornwinder2d does — the perturbation
    # must be O(eps) so it does not put an error floor on the shape functions.
    xc = copy(x)
    xc[:, 3] .= min.(1 - 1e-13, xc[:, 3])
    s23 = min.(xc[:, 2] .+ xc[:, 3], -1e-13)

    a = @. -2.0 * (1.0 + xc[:, 1]) / s23 - 1.0
    b = @. 2.0 * (1.0 + xc[:, 2]) / (1.0 - xc[:, 3]) - 1.0
    c = xc[:, 3]

    # Derivatives of the collapsed map (w.r.t. the bi-unit coordinates)
    da_dx = @. -2.0 / s23
    da_dy = @. 2.0 * (1.0 + xc[:, 1]) / s23^2
    da_dz = da_dy
    db_dy = @. 2.0 / (1.0 - xc[:, 3])
    db_dz = @. 2.0 * (1.0 + xc[:, 2]) / (1.0 - xc[:, 3])^2

    for i in 1:npol
        p_order, q_order, r_order = pqr[i, 1], pqr[i, 2], pqr[i, 3]

        pp = poly_jacobi(p_order, 0, 0)
        qp = poly_jacobi(q_order, 2 * p_order + 1, 0)
        rp = poly_jacobi(r_order, 2 * p_order + 2 * q_order + 2, 0)

        # fold the collapsed-coordinate factors ((1-b)/2)^p and ((1-c)/2)^(p+q)
        # into the b- and c-polynomials
        for _ in 1:p_order
            qp *= Polynomial([0.5, -0.5])
        end
        for _ in 1:(p_order + q_order)
            rp *= Polynomial([0.5, -0.5])
        end

        dpp = derivative(pp)
        dqp = derivative(qp)
        drp = derivative(rp)

        pval, qval, rval = pp.(a), qp.(b), rp.(c)
        dpval, dqval, drval = dpp.(a), dqp.(b), drp.(c)

        fc = sqrt(2.0 * (2.0 * p_order + 1.0) * (p_order + q_order + 1.0) *
                  (2.0 * (p_order + q_order + r_order) + 3.0))

        f[:, i] .= fc .* pval .* qval .* rval
        fx[:, i] .= fc .* dpval .* qval .* rval .* da_dx
        fy[:, i] .= fc .* (dpval .* qval .* rval .* da_dy .+
                           pval .* dqval .* rval .* db_dy)
        fz[:, i] .= fc .* (dpval .* qval .* rval .* da_dz .+
                           pval .* dqval .* rval .* db_dz .+
                           pval .* qval .* drval)
    end

    # chain rule of the unit -> bi-unit transform
    fx .*= 2.0
    fy .*= 2.0
    fz .*= 2.0

    return f, fx, fy, fz
end

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
    # Adjust second coordinate (column 2 in Julia) to avoid the 0/0 in the
    # collapsed-coordinate map at the top vertex (where 1 + x1 = 0, so e1
    # evaluates to exactly -1 as required).  The offset perturbs the Jacobi
    # factor q(x2) by O(eps), so it must be at roundoff level: the previous
    # value of 1e-8 put an O(1e-8) error floor on all shape functions.
    xc[:, 2] .= min.(1 - 1e-13, xc[:, 2])
    
    # Set up the evaluation coordinates e.
    # In Python, e[:,0] corresponds to the first column; in Julia, we use column 1.
    e = zeros(Float64, size(xc))
    e[:, 1] .= 2 .* (1.0 .+ xc[:, 1]) ./ (1 .- xc[:, 2]) .- 1.0
    e[:, 2] .= xc[:, 2]
    
    # For points where the original x's second coordinate equals 1,
    # set e accordingly (Python: e[ii,0]=-1, e[ii,1]=1; Julia: columns 1 and 2).
    # comment this out in the Julia code, else it creates singularity!
    # idx = findall(x[:, 2] .== 1.0)
    # for i in idx
    #     e[i, 1] = -1.0
    #     e[i, 2] = 1.0
    # end
    
    # Build the Vandermonde matrix for the Koornwinder polynomials
    for i in 1:npol
        p_order = pq[i, 1]  # corresponds to pq[ii,0] in Python
        q_order = pq[i, 2]  # corresponds to pq[ii,1] in Python
        # Obtain Jacobi polynomial coefficients for p and q parts.
        pp = poly_jacobi(p_order, 0, 0)
        qp = poly_jacobi(q_order, 2 * p_order + 1, 0)
        
        # Multiply by the collapsed-coordinate factor ((1 - x) / 2)^p_order
        for j in 1:p_order
            qp *= Polynomial([0.5, -0.5])
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
            qp *= Polynomial([0.5, -0.5])
        end

        # Compute derivative polynomials.
        dpp = derivative(pp)
        dqp = derivative(qp)

        # Evaluate polynomials and their derivatives.
        pval  = pp.(e[:, 1])
        qval  = qp.(e[:, 2])
        dpval = dpp.(e[:, 1])
        dqval = dqp.(e[:, 2])
        
        fc = sqrt((2.0 * p_order + 1.0) * 2.0 * (p_order + q_order + 1.0))

        fx[:, i] .= fc .* dpval .* qval .* de1[:, 1]
        fy[:, i] .= fc .* (dpval .* qval .* de1[:, 2] .+ pval .* dqval)
    end
    
    # Adjust derivatives by a factor of 2.
    fx .*= 2.0
    fy .*= 2.0
    
    return f, fx, fy
end