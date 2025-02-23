"""
    newton_raphson(f, fgrad, x₀; abstol=1e-8, reltol=1e-8, maxiter=100)
Newton-Raphson method for root finding.
"""
function newton_raphson(f, fgrad, x₀; abstol=1e-8, reltol=1e-8, maxiter=100)
    x = x₀
    for _ in 1:maxiter
        x′ = x - f(x) / fgrad(x)
        residual_norm = abs(f(x))
        tolerance = max(reltol * residual_norm, abstol)
        if residual_norm < tolerance
            return x′
        end
        x = x′
    end
    return x
end