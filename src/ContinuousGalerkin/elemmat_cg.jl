"""
    elemmat_cg(pcg, master, source, param) -> (A, F)

Elemental stiffness matrix and force vector of the continuous-Galerkin
convection-diffusion(-reaction) operator — the per-element reference
implementation the batched path ([`cg_element_system`](@ref)) is verified
against. Dimension-generic (triangles and tetrahedra).

- `pcg (npl, Dim)`: element node coordinates
- `master`: reference element
- `source`: forcing function, called with the coordinates splatted
  (`source(x, y)` in 2D, `source(x, y, z)` in 3D)
- `param`: named tuple with diffusivity `κ`, convective velocity `c`
  (length `Dim`), and reaction coefficient `s`

Returns the local element matrix `A (npl, npl)` and force vector `F (npl)`.
"""
function elemmat_cg(pcg, master, source, param)
    Dim = ndims(master)
    npl = size(pcg, 1)
    κ, c, s = param.κ, param.c, param.s

    A = zeros(npl, npl)
    F = zeros(npl)
    for k in eachindex(master.gwgh)
        ϕ = @view master.shap[:, 1, k]
        dϕ = @view master.shap[:, 2:(Dim + 1), k]   # reference derivatives (npl, Dim)

        # isoparametric map: J[k, d] = ∂x_d/∂ξ_k, invJ[d, k] = ∂ξ_k/∂x_d
        J = dϕ' * pcg
        detJ = det(J)
        invJ = inv(J)
        ∇ϕ = dϕ * invJ'                             # ∇ϕ[i, d] = ∂ϕᵢ/∂x_d
        w = detJ * master.gwgh[k]

        # κ∫∇ϕᵢ·∇ϕⱼ  −  ∫(c·∇ϕᵢ)ϕⱼ  +  s∫ϕᵢϕⱼ
        for j in 1:npl, i in 1:npl
            diff = 0.0
            conv = 0.0
            for d in 1:Dim
                diff += ∇ϕ[i, d] * ∇ϕ[j, d]
                conv += ∇ϕ[i, d] * c[d]
            end
            A[i, j] += (κ * diff - conv * ϕ[j] + s * ϕ[i] * ϕ[j]) * w
        end

        # ∫f·ϕᵢ with f at the quadrature point
        x = vec(ϕ' * pcg)
        F .+= ϕ .* (w * source(x...))
    end
    return A, F
end
