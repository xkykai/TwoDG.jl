# Numerical (surface) fluxes as swappable callable objects, decoupled from
# the equations (Trixi's `surface_flux` pattern). A numerical flux is any
# callable `(eq, uL, uR, n, x, t) -> SVector{nc}`; the built-ins dispatch on
# the equation type, and a user flux needs no package edit:
#
#     struct MyFlux end
#     (::MyFlux)(eq, uL, uR, n, x, t) = ...
#     prob = DGProblem(eq, mesh; numerical_flux = MyFlux(), ...)

"""
    LaxFriedrichs()

Local Lax–Friedrichs (Rusanov) flux,

    F̂ = ½ (F(uL) + F(uR)) ⋅ n − ½ λ (uR − uL),   λ = max(|λ(uL)|, |λ(uR)|),

using the equation's [`flux`](@ref) and [`max_abs_speed`](@ref). Works for
any equation that implements those two methods; exact upwinding for linear
scalar convection.
"""
struct LaxFriedrichs end

@inline function (::LaxFriedrichs)(eq::AbstractEquation, uL, uR, n, x, t)
    central = (normal_flux(eq, uL, n, x, t) + normal_flux(eq, uR, n, x, t)) / 2
    λ = max(max_abs_speed(eq, uL, n, x, t), max_abs_speed(eq, uR, n, x, t))
    return central - λ / 2 * (uR - uL)
end

"""
    RoeFlux()

Roe's approximate Riemann solver (Roe, JCP 43:357, 1981),

    F̂ = ½ (F(uL) + F(uR)) ⋅ n − ½ |Â(ũ)| (uR − uL),

with `Â` the flux Jacobian evaluated at the Roe average `ũ`. Implemented for
[`EulerEquations`](@ref) (density-weighted Roe average) and
[`WaveEquation`](@ref) (exact characteristic decomposition of the linear
system).
"""
struct RoeFlux end

# per-equation methods live next to their equations (euler.jl, wave.jl)
