using TwoDG.Meshes: mkmesh_trefftz
using TwoDG.Masters: Master, shape2d, get_local_face_nodes
using Dierckx

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
    x0 = tparam[1]
    y0 = tparam[2]
    n = tparam[3]
    
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
    wd = ((1 .+ zd) ./ (1 .- zd)) .* n
    
    # Find leading edge point
    xle = minimum(real.(wd))
    ii = argmin(real.(wd))

    # Create cubic splines for the camber line
    x_sup = reverse(real.(wd[1:ii]))
    y_sup = reverse(imag.(wd[1:ii]))
    x_slo = real.(wd[ii:end])
    y_slo = imag.(wd[ii:end])

    sup = Spline1D(x_sup, y_sup, k=3)
    slo = Spline1D(x_slo, y_slo, k=3)

    # Create lookup functions that map from x to y for the camber line
    function sup_func(x)
        # Find normalized position
        idx = (x - minimum(x_sup)) / (maximum(x_sup) - minimum(x_sup))
        # Clamp to valid range
        idx = clamp(idx, 0.0, 1.0)
        return sup(idx)
    end

    function slo_func(x)
        # Find normalized position
        idx = (x - minimum(x_slo)) / (maximum(x_slo) - minimum(x_slo))
        # Clamp to valid range
        idx = clamp(idx, 0.0, 1.0)
        return slo(idx)
    end
    
    # K-T inverse mapping
    A = (z .- n) ./ (z .+ n)
    anga = angle.(A)
    
    # Create masks for regions where phase adjustment is needed
    zr = real.(z)
    zi = imag.(z)
    
    # Pre-compute the middle of camber line
    midpts = [0.5 * (sup_func(zr[i]) + slo_func(zr[i])) for i in eachindex(zr) if zr[i] > xle && zr[i] <= n]
    
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

function trefftz(V∞, α, m=15, n=30, porder=3, node_spacing_type=0, tparam=[0.1, 0.05, 1.98])
    # Generate mesh for Trefftz airfoil with given parameters
    mesh = mkmesh_trefftz(m, n, porder, node_spacing_type, tparam)
    master = Master(mesh)  # Create master element for reference transformations
    dgnodes = mesh.dgnodes  # Nodal coordinates of the discontinuous Galerkin mesh
   
    # Calculate analytical potential flow solution
    ψ, vx_analytical, vy_analytical, Γ_analytical = potential_trefftz(mesh.dgnodes[:, 1, :], mesh.dgnodes[:, 2, :], V=V∞, alpha=α, tparam=tparam)
    xs_foil, ys_foil, chord = trefftz_points(tparam)  # Get airfoil coordinates and chord length
    shapefunction_local = shape2d(porder, master.plocal, master.plocal[:, 2:3])  # Shape functions for numerical integration
    vx = zeros(size(ψ))  # Initialize velocity arrays
    vy = zeros(size(ψ))
    
    # Calculate velocity field from potential gradient
    for i in axes(dgnodes, 3)  # Loop over all elements
        element_coordinates = @view dgnodes[:, :, i]  # Coordinates of current element
        for j in axes(master.plocal, 1)  # Loop over all quadrature points
            # Calculate Jacobian matrix for coordinate transformation (physical to reference element)
            J = shapefunction_local[:, 2:3, j]' * element_coordinates
            invJ = inv(J)  # Inverse Jacobian for gradient transformation
            
            # Calculate gradient of potential (∂ψ/∂x, ∂ψ/∂y) using chain rule via Jacobian
            ∂ψ = sum([ψ[k, i] .* (invJ * shapefunction_local[k, 2:3, j]) for k in axes(master.plocal, 1)])
            
            # Velocity components (negative gradient of potential)
            vx[j, i] = -∂ψ[2]  # u = -∂ψ/∂y
            vy[j, i] = ∂ψ[1]   # v = ∂ψ/∂x
        end
    end
    
    # Calculate circulation Γ around the airfoil by integrating along the outer boundary
    # Circulation is directly related to lift via Kutta-Joukowski theorem
    Γ = 0
    boundary_facenumbers = findall(x -> x == -2, mesh.f[:, 4])  # Identify outer boundary faces
    for face_number in boundary_facenumbers
        it = mesh.f[face_number, 3]  # Element number containing this face
        node_numbers = get_local_face_nodes(mesh, master, face_number)  # Get nodes on this face
        element_coordinates = @view dgnodes[node_numbers, :, it]  # Coordinates of these nodes
        
        for j in eachindex(master.gw1d)  # Loop over 1D quadrature points on the face
            τ = master.sh1d[:, 2, j]' * element_coordinates  # Tangent vector at quadrature point
            
            # Interpolate velocity at quadrature point
            vxj = sum(vx[node_numbers, it] .* master.sh1d[:, 1, j])
            vyj = sum(vy[node_numbers, it] .* master.sh1d[:, 1, j])
            
            # Add contribution to circulation (v·τ*ds)
            Γ += sum(master.gw1d[j] * dot(vcat(vxj, vyj), τ))
        end
    end
    
    # Calculate lift coefficients using Kutta-Joukowski theorem (CL = -2Γ/(V∞*chord))
    CL = -2 * Γ / V∞ / chord
    CL_analytical = -2 * Γ_analytical / V∞ / chord
    
    # Calculate pressure coefficient using Bernoulli's equation (Cp = 1 - (v²/V∞²))
    CP = 1 .- (vx .^2 .+ vy .^2) ./ V∞^2
    CP_analytical = 1 .- (vx_analytical .^2 .+ vy_analytical .^2) ./ V∞^2
    
    # Calculate force coefficients by integrating pressure along the airfoil surface
    Clift = 0
    CF = zeros(2)  # Force vector [Fx, Fy]
    airfoil_facenumbers = findall(x -> x == -1, mesh.f[:, 4])  # Identify airfoil surface faces
    
    for face_number in airfoil_facenumbers
        it = mesh.f[face_number, 3]  # Element containing this face
        node_numbers = get_local_face_nodes(mesh, master, face_number)
        element_coordinates = @view dgnodes[node_numbers, :, it]
        
        for j in eachindex(master.gw1d)  # Loop over quadrature points
            τ = master.sh1d[:, 2, j]' * element_coordinates  # Tangent vector at quadrature point
            n = [τ[2], -τ[1]]  # Normal vector (90° rotation of tangent)
            
            # Interpolate pressure coefficient at quadrature point
            CPj = sum(CP[node_numbers, it] .* master.sh1d[:, 1, j])
            
            # Add contribution to force (Cp*n*ds)
            CF .+= master.gw1d[j] * CPj .* n
        end
    end
    CF /= chord  # Non-dimensionalize by chord length
    
    # Transform force components to lift and drag in the freestream coordinate system
    u∞_direction = [cos(deg2rad(α)), sin(deg2rad(α)), 0]  # Freestream direction vector
    
    # Lift is perpendicular to freestream direction (cross product)
    Clift = (u∞_direction × vcat(CF, 0))[3]
    
    # Drag is parallel to freestream direction (dot product)
    Cdrag = -vcat(CF, 0) ⋅ u∞_direction
    
    # Calculate moment coefficient around quarter-chord point
    CM = 0
    quarter_chord_coordinate = [minimum(xs_foil) + chord/4, 0, 0]  # Quarter-chord location
    
    for face_number in airfoil_facenumbers
        it = mesh.f[face_number, 3]
        node_numbers = get_local_face_nodes(mesh, master, face_number)
        element_coordinates = @view dgnodes[node_numbers, :, it]
        
        for j in eachindex(master.gw1d)
            τ = master.sh1d[:, 2, j]' * element_coordinates  # Tangent vector
            n = [τ[2], -τ[1], 0]  # Normal vector in 3D
            CPj = sum(CP[node_numbers, it] .* master.sh1d[:, 1, j])  # Interpolated pressure
            
            # Calculate position of quadrature point
            gauss_coordinate = vec(master.sh1d[:, 1, j]' * element_coordinates)
            
            # Calculate moment arm × normal force (r × F)
            moment = ((vcat(gauss_coordinate, 0) .- quarter_chord_coordinate) × n)[3]
            
            # Add contribution to moment coefficient
            CM += master.gw1d[j] * CPj * moment
        end
    end
    CM /= chord^2  # Non-dimensionalize by chord²
    
    return mesh, master, xs_foil, ys_foil, chord, ψ, vx, vy, Γ, CP, CF, Clift, Cdrag, CM, vx_analytical, vy_analytical, Γ_analytical, CP_analytical, CL, CL_analytical
end