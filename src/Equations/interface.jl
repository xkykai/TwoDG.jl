# The dispatch contract every equation implements. These generic functions —
# not any struct — are the physics API: the DG/HDG/CG kernels call them, and a
# user adds an equation by defining methods on their own type.

"""
    AbstractEquation

Supertype of all equations. A concrete equation is a small immutable struct
holding its physical parameters as typed fields (e.g. `EulerEquations(γ)`),
and implements the dispatch contract:

| method | required for | meaning |
|---|---|---|
| [`nvariables`](@ref)`(eq)` | all | number of conserved components |
| [`varnames`](@ref)`(eq)` | all | component names, for output |
| [`flux`](@ref)`(eq, u, x, t)` | all | physical flux `(fx, fy)` |
| [`max_abs_speed`](@ref)`(eq, u, n, x, t)` | Lax–Friedrichs-type fluxes | max characteristic speed along `n` |
| [`default_numerical_flux`](@ref)`(eq)` | `solve` defaults | numerical flux used when none is given |
| [`has_diffusion`](@ref)`(eq)` | viscous (LDG) terms | `true` enables the gradient/viscous path |
| [`viscous_flux`](@ref)`(eq, u, q, x, t)` | if `has_diffusion` | viscous physical flux `(fx, fy)` |

States `u` are `SVector{nc}`, positions `x` are `SVector{2}`, gradients `q`
are `SMatrix{2, nc}` (rows = x/y derivative). All methods must be `@inline`,
allocation-free, and generic in the element type `T` so they compile inside
CPU/GPU kernels at any precision.
"""
abstract type AbstractEquation end

"""
    nvariables(eq) -> Int

Number of conserved components of an equation (or of a physics bundle that
wraps one).
"""
function nvariables end

"""
    varnames(eq) -> NTuple{nc, Symbol}

Names of the conserved components, in storage order.
"""
function varnames end

"""
    flux(eq, u, x, t) -> (fx, fy)

Physical (volume) flux of the equation at state `u::SVector{nc}` and position
`x::SVector{2}`: the pair of `SVector{nc}` flux components such that the
conservation law reads `∂u/∂t + ∂fx/∂x + ∂fy/∂y = source`. Defined **once**
per equation; the numerical fluxes, boundary fluxes, and volume terms all
call it.
"""
function flux end

"""
    normal_flux(eq, u, n, x, t) -> SVector{nc}

Physical flux projected on the unit normal `n`: `flux(eq, u, x, t) ⋅ n`.
Numerical fluxes use it for their central part.
"""
@inline function normal_flux(eq::AbstractEquation, u, n, x, t)
    fx, fy = flux(eq, u, x, t)
    return fx * n[1] + fy * n[2]
end

"""
    max_abs_speed(eq, u, n, x, t) -> Real

Maximum absolute characteristic speed of `eq` at state `u` in direction `n`
(e.g. `|v ⋅ n|` for convection, `|u ⋅ n| + c` for Euler). Used by
[`LaxFriedrichs`](@ref)-type dissipation and CFL estimates.
"""
function max_abs_speed end

"""
    has_diffusion(eq) -> Bool

Whether the equation has second-order (viscous/diffusive) terms, i.e. whether
the LDG gradient and viscous-flux path must run. Defaults to `false`.
"""
has_diffusion(::AbstractEquation) = false

"""
    viscous_flux(eq, u, q, x, t) -> (fx, fy)

Viscous physical flux at state `u` and gradient `q::SMatrix{2, nc}` (rows are
the x/y derivatives). Only needed when [`has_diffusion`](@ref) is `true`.
"""
function viscous_flux end

"""
    default_numerical_flux(eq)

The numerical flux `solve` uses when the problem does not specify one
(e.g. `RoeFlux()` for [`EulerEquations`](@ref), [`LaxFriedrichs`](@ref) for
scalar convection). Any callable `(eq, uL, uR, n, x, t) -> SVector{nc}` can
replace it per problem.
"""
function default_numerical_flux end

"""
    default_stabilization(eq)

The viscous (LDG) stabilization `solve` uses when the problem does not
specify one; `nothing` for purely hyperbolic equations.
"""
default_stabilization(::AbstractEquation) = nothing
