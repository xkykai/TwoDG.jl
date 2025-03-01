"""
    newton_raphson(f, fgrad, x₀; abstol=1e-8, reltol=1e-8, maxiter=100)
Newton-Raphson method for root finding.
"""
function newton_raphson(f, fgrad, x₀; abstol=1e-8, reltol=1e-8, maxiter=100)
    # Initialize with the starting point
    x = x₀
    for _ in 1:maxiter
        # Newton-Raphson update formula: x_{n+1} = x_n - f(x_n)/f'(x_n)
        x′ = x - f(x) / fgrad(x)
        
        # Calculate the absolute value of f(x) to measure how close we are to the root
        residual_norm = abs(f(x))
        
        # Dynamic tolerance threshold - uses both relative and absolute tolerance
        # This allows appropriate scaling for both large and small values
        tolerance = max(reltol * residual_norm, abstol)
        
        # Check if we've converged to a root within tolerance
        if residual_norm < tolerance
            return x′  # Return the new value if converged
        end
        
        # Update x for the next iteration
        x = x′
    end
    
    # Return the final approximation if max iterations reached without convergence
    return x
end