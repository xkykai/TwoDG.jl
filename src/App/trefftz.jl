using Interpolations

function trefftz_points(tparam=[0.1, 0.05, 1.98], np=120)
    # Extract parameters
    x0, y0, n = tparam
    
    # K-T Transform
    cc = complex(-x0, y0)
    th = range(0, 2π, length=np+1)[1:end-1]  # Avoid duplicating the endpoint
    
    # Using broadcasting for vectorized operations
    xc = (1 - cc) .* exp.(im .* th)
    wd = cc .+ xc
    zd = ((wd .- 1) ./ (wd .+ 1)) .^ n
    wd = ((1 .+ zd) ./ (1 .- zd)) .* n
    
    # Extract real and imaginary parts
    x = real.(wd)
    y = imag.(wd)
    
    # Calculate chord
    xle = minimum(real.(wd))
    chord = n - xle
    
    return x, y, chord
end

function potential_trefftz(x, y; V=1.0, alpha=0.0, tparam=[0.1, 0.05, 1.98])
    """
    potential_trefftz calculates the 2d potential flow for trefftz airfoil.
    
    Returns:
        psi:       value of stream function at input points
        velx:      x component of the velocity at input points
        vely:      y component of the velocity at input points
        gamma:     circultation. lift force= v*gamma
    
    Parameters:
        x:         x coordinates of input points
        y:         y coordinates of input points
        V:         free stream velocity magnitude (default=1)
        alpha:     angle of attack in degrees (default=0)
        tparam:    trefftz foil parameters
                  tparam[1] = left x-shift of circle center
                              (trailing edge at (1,0)). (default=0.1)
                  tparam[2] = y-shift of circle center. (default=0.05)
                  tparam[3] = k-t exponent (=< 2) (2:jukowski). (default=1.98)
    """
    # Extract parameters
    x0 = Float64(tparam[1])
    y0 = Float64(tparam[2])
    n = Float64(tparam[3])
    
    # First Rotate to ensure that a point stays at the trailing edge
    rot = atan(y0, 1+x0)
    r = sqrt((1+x0)^2 + y0^2)
    
    # Initialize complex vectors
    z = complex.(x, y)
    
    # First calculate an approximate camber line
    cc = complex(-x0, y0)
    th = range(0, 2π, length=121)
    xc = (1-cc) .* exp.(im .* th)
    wd = cc .+ xc
    zd = ((wd .- 1) ./ (wd .+ 1)) .^ n
    wd = ((1 .+ zd) ./ (1 .- zd))
    
    # Find leading edge point
    xle = minimum(real.(wd))
    ii = argmin(real.(wd))
    
    # Create cubic splines for the camber line
    sup = CubicSplineInterpolation(reverse(real.(wd[1:ii])), reverse(imag.(wd[1:ii])))
    slo = CubicSplineInterpolation(real.(wd[ii:end]), imag.(wd[ii:end]))
    
    # K-T inverse mapping
    A = (z .- n) ./ (z .+ n)
    anga = angle.(A)
    
    # Create masks for regions where phase adjustment is needed
    zr = real.(z)
    zi = imag.(z)
    
    # Pre-compute the middle of camber line
    midpts = [0.5 * (sup(zr[i]) + slo(zr[i])) for i in eachindex(zr) if zr[i] > xle && zr[i] <= n]
    
    # Apply phase corrections
    for i in eachindex(z)
        if zr[i] > xle && zr[i] <= n
            midpt = 0.5 * (sup(zr[i]) + slo(zr[i]))
            if zi[i] < midpt && anga[i] > 1.5
                anga[i] -= 2.0π
            elseif zi[i] > midpt && anga[i] < -1.5
                anga[i] += 2.0π
            end
        end
    end
    
    # Calculate B and v
    B = abs.(A) .^ (1.0/n) .* exp.(im .* anga ./ n)
    v = (1.0 .+ B) ./ (1.0 .- B)
    
    # Scale back
    w = (1.0/r) .* exp(im*rot) .* (v .+ x0 .- im .* y0)
    
    # Effective angle of attack
    alphef = π * alpha/180.0 + rot
    
    # Calculate circulation
    dphidw = -V .* (exp(-im*alphef) .- 1.0 ./ exp(-im*alphef))
    dvortdw = im / (2π)
    Gamma = -real(dphidw/dvortdw)
    
    # Calculate potential
    phi = -V .* r .* (w .* exp(-im*alphef) .+ 1.0 ./ (w .* exp(-im*alphef)))
    vort = im .* Gamma .* log.(w) ./ (2π)
    psi = imag.(phi .+ vort)
    
    # Handle trailing edge
    for i in eachindex(w)
        if abs(w[i] - 1) < 1.0e-6
            w[i] = complex(2.0, 0.0)
        end
    end
    
    # Calculate velocity components
    dphidw = -V .* r .* (exp(-im*alphef) .- 1.0 ./ (w .* w .* exp(-im*alphef)))
    dvortdw = im .* Gamma ./ (2π .* w)
    dwdv = (1/r) .* exp(im*rot)
    dvdB = 2.0 ./ (1.0 .- B) .^ 2
    
    # Calculate auxiliary values for the derivative
    aux = abs.(A)
    for i in eachindex(aux)
        if aux[i] > 1.0e-12
            aux[i] = aux[i]^((1.0-n)/n)
        end
    end
    
    dBdz = (1.0/n) .* aux .* exp.(im .* anga .* (1.0-n)/n) .* (1 .- A) ./ (z .+ n)
    dphi = (dphidw .+ dvortdw) .* dwdv .* dvdB .* dBdz
    
    # Set unbounded derivatives at trailing edge to zero
    for i in eachindex(dphi)
        if abs(w[i] - 2.0) < 1.0e-6
            dphi[i] = complex(0.0, 0.0)
        end
    end
    
    # Extract velocity components
    velx = -real.(dphi)
    vely = imag.(dphi)
    
    return psi, velx, vely, Gamma
end