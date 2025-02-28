using Interpolations
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
    # Generate mesh
    mesh = mkmesh_trefftz(m, n, porder, node_spacing_type, tparam)
    master = Master(mesh)
    dgnodes = mesh.dgnodes
    
    # Calculate potential
    ψ, vx_analytical, vy_analytical, Γ_analytical = potential_trefftz(mesh.dgnodes[:, 1, :], mesh.dgnodes[:, 2, :], V=V∞, alpha=α, tparam=tparam)

    _, _, chord = trefftz_points(tparam)

    shapefunction_local = shape2d(porder, master.plocal, master.plocal[:, 2:3])

    vx = zeros(size(ψ))
    vy = zeros(size(ψ))

    for i in axes(dgnodes, 3)
        element_coordinates = @view dgnodes[:, :, i]

        for j in axes(master.plocal, 1)
            J = shapefunction_local[:, 2:3, j]' * element_coordinates
            invJ = inv(J)

            ∂ψ = sum([ψ[k, i] .* (invJ * shapefunction_local[k, 2:3, j]) for k in axes(master.plocal, 1)])

            vx[j, i] = -∂ψ[2]
            vy[j, i] = ∂ψ[1]
        end
    end

    # vx_singularity_locations = findall(x -> abs(x) > 2, vx)
    # vy_singularity_locations = findall(x -> abs(x) > 2, vy)

    # for i in vx_singularity_locations
    #     vx[i] = 0
    # end

    # for i in vy_singularity_locations
    #     vy[i] = 0
    # end

    Γ = 0
    boundary_facenumbers = findall(x -> x == -2, mesh.f[:, 4])
    for face_number in boundary_facenumbers
        it = mesh.f[face_number, 3]
        node_numbers = get_local_face_nodes(mesh, master, face_number)
        element_coordinates = @view dgnodes[node_numbers, :, it]

        for j in eachindex(master.gw1d)
            τ = master.sh1d[:, 2, j]' * element_coordinates

            vxj = sum(vx[node_numbers, it] .* master.sh1d[:, 1, j])
            vyj = sum(vy[node_numbers, it] .* master.sh1d[:, 1, j])

            Γ += sum(master.gw1d[j] * dot(vcat(vxj, vyj), τ))
        end
    end

    CL = -2 * Γ / V∞ / chord
    CP = 1 .- (vx .^2 .+ vy .^2) ./ V∞^2
    CP_analytical = 1 .- (vx_analytical .^2 .+ vy_analytical .^2) ./ V∞^2

    Clift = 0
    CF = zeros(2)
    airfoil_facenumbers = findall(x -> x == -1, mesh.f[:, 4])
    # u∞_direction = [cos(deg2rad(α)), sin(deg2rad(α))]
    u∞_direction = [cos(deg2rad(α)), sin(deg2rad(α)), 0]
    for face_number in airfoil_facenumbers
        it = mesh.f[face_number, 3]
        node_numbers = get_local_face_nodes(mesh, master, face_number)
        element_coordinates = @view dgnodes[node_numbers, :, it]

        for j in eachindex(master.gw1d)
            τ = master.sh1d[:, 2, j]' * element_coordinates
            n = [τ[2], -τ[1]]
            # @info "n = $(n ./ norm(n))"
            # n = [τ[2], -τ[1], 0]

            CPj = sum(CP[node_numbers, it] .* master.sh1d[:, 1, j])

            # cosθ = n ⋅ u∞_direction
            # cosθ = (n × u∞_direction)[3]

            CF .+= master.gw1d[j] * CPj .* n
        end
    end

    CF /= chord

    CF_angle = atan(CF[2], CF[1])
    Clift = norm(CF) * sin(CF_angle - deg2rad(α))
    Cdrag = norm(CF) * cos(CF_angle - deg2rad(α))

    CM = 0
    x_quarter_chord = [0.25 * chord, 0, 0]
    for face_number in airfoil_facenumbers
        it = mesh.f[face_number, 3]
        node_numbers = get_local_face_nodes(mesh, master, face_number)
        element_coordinates = @view dgnodes[node_numbers, :, it]

        for j in eachindex(master.gw1d)
            τ = master.sh1d[:, 2, j]' * element_coordinates
            n = [τ[2], -τ[1], 0]

            CPj = sum(CP[node_numbers, it] .* master.sh1d[:, 1, j])
            gauss_coordinate = vec(master.sh1d[:, 1, j]' * element_coordinates)
            moment = ((vcat(gauss_coordinate, 0) .- x_quarter_chord) × n)[3]

            # cosθ = n ⋅ u∞_direction
            # cosθ = (n × u∞_direction)[3]

            CM += master.gw1d[j] * CPj * moment
        end
    end

    CM /= chord^2

    return mesh, master, ψ, vx, vy, Γ, CL, CP, CP_analytical, Clift, Cdrag, CF, CM, vx_analytical, vy_analytical, Γ_analytical
end