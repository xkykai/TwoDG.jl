# Derived-quantity utilities for the Euler equations. The legacy matrix-flux
# functions (euleri_roe/eulerb/eulerv) were retired in favor of the pointwise
# implementation in pointwise.jl (see `mkapp_euler_pt`).

"""
eulereval calculates derived quantities for the euler equation variables.

   u[npl,4,nt]:   states
   str:           string used to specify requested quantity
                  - str: "r" density
                  - str: "u" u_x velocity
                  - str: "v" u_y velocity
                  - str: "p" pressure
                  - str: "c" speed of sound
                  - str: "Jp" characteristic variable J+
                  - str: "Jm" characteristic variable J-
                  - str: "M" Mach number
                  - str: "s" entropy
   gam:           value of gamma
Returns:
   sca[npl,nt]:   scalar field requested by str
"""
function eulereval(u::Array{T,3}, str, gam::T) where T<:AbstractFloat
    npl, _, nt = size(u)
    
    if str == "r"
        # Density - just return a view of the first component
        return view(u, :, 1, :)
    end
    
    # Pre-allocate output array for all other cases
    sca = Array{T}(undef, npl, nt)
    
    if str == "u"
        # X-velocity
        @inbounds for j in 1:nt, i in 1:npl
            sca[i, j] = u[i, 2, j] / u[i, 1, j]
        end
    elseif str == "v"
        # Y-velocity
        @inbounds for j in 1:nt, i in 1:npl
            sca[i, j] = u[i, 3, j] / u[i, 1, j]
        end
    else
        # For other quantities, we need the velocity components
        uv = similar(sca)
        vv = similar(sca)
        @inbounds for j in 1:nt, i in 1:npl
            uv[i, j] = u[i, 2, j] / u[i, 1, j]
            vv[i, j] = u[i, 3, j] / u[i, 1, j]
        end
        
        if str == "p"
            # Pressure
            @inbounds for j in 1:nt, i in 1:npl
                sca[i, j] = (gam - 1) * (u[i, 4, j] - 0.5 * (u[i, 2, j] * uv[i, j] + u[i, 3, j] * vv[i, j]))
            end
        else
            # Calculate pressure for other quantities
            p = similar(sca)
            @inbounds for j in 1:nt, i in 1:npl
                p[i, j] = (gam - 1) * (u[i, 4, j] - 0.5 * (u[i, 2, j] * uv[i, j] + u[i, 3, j] * vv[i, j]))
            end
            
            if str == "c"
                # Speed of sound
                @inbounds for j in 1:nt, i in 1:npl
                    sca[i, j] = sqrt(gam * p[i, j] / u[i, 1, j])
                end
            elseif str == "Jp" || str == "Jm"
                # Characteristic variables J+ and J-
                c = similar(sca)
                @inbounds for j in 1:nt, i in 1:npl
                    c[i, j] = sqrt(gam * p[i, j] / u[i, 1, j])
                end
                
                if str == "Jp"
                    @inbounds for j in 1:nt, i in 1:npl
                        sca[i, j] = u[i, 2, j] + 2 * c[i, j] / (gam - 1)
                    end
                else  # str == "Jm"
                    @inbounds for j in 1:nt, i in 1:npl
                        sca[i, j] = u[i, 2, j] - 2 * c[i, j] / (gam - 1)
                    end
                end
            elseif str == "M"
                # Mach number
                @inbounds for j in 1:nt, i in 1:npl
                    u2 = sqrt(uv[i, j]^2 + vv[i, j]^2)
                    sca[i, j] = u2 / sqrt(gam * p[i, j] / u[i, 1, j])
                end
            elseif str == "s"
                # Entropy
                @inbounds for j in 1:nt, i in 1:npl
                    sca[i, j] = p[i, j] / (u[i, 1, j]^gam)
                end
            else
                error("Unknown quantity: $str")
            end
        end
    end
    
    return sca
end

function riemann_to_canonical(v, s, J⁺, J⁻, γ)
    c = (γ - 1) / 4 * (J⁺ - J⁻)
    ρu₁ = (J⁺ + J⁻) / 2
    ρ = (c^2 / γ / s)^(1 / (γ - 1))
    ρu₂ = ρ * v
    p = s * ρ^γ
    ρE = p / (γ - 1) + 0.5 * (ρu₁^2 + ρu₂^2) / ρ
    return ρ, ρu₁, ρu₂, ρE
end

function canonical_to_riemann(ρ, ρu₁, ρu₂, ρE, γ)
    p = (γ - 1) * (ρE - 0.5 * (ρu₁^2 + ρu₂^2) / ρ)
    v = ρu₂ / ρ
    s = p / (ρ^γ)
    c = sqrt(γ * p / ρ)
    J⁺ = ρu₁ + 2c / (γ - 1)
    J⁻ = ρu₁ - 2c / (γ - 1)
    return v, s, J⁺, J⁻
end