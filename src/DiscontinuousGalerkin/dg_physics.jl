# The compiled physics bundle the DG kernels consume: an equation, a
# numerical flux, one boundary condition per boundary tag, and optional
# source/stabilization — all small immutable objects selected by dispatch
# (REFACTOR_PLAN Phase 4; replaces the `App` callback-slot struct).

using Adapt
using ..Equations
using ..Equations: apply_boundary_flux, apply_boundary_trace,
                   apply_boundary_viscous_flux
import ..Equations: nvariables, has_diffusion

"""
    DGPhysics(equation; boundary_conditions,
              numerical_flux=default_numerical_flux(equation),
              source=nothing,
              stabilization=default_stabilization(equation))

Everything the DG residual kernels need to know about the physics:

- `equation :: AbstractEquation` — implements `flux`/`nvariables`/….
- `boundary_conditions` — one [`BoundaryCondition`](@ref TwoDG.Equations.BoundaryCondition)
  per boundary tag (any collection; stored as a `Tuple` so kernels dispatch
  statically per boundary).
- `numerical_flux` — any callable `(eq, uL, uR, n, x, t) -> SVector`.
- `source` — `nothing` or a callable `(u, x, t) -> SVector`.
- `stabilization` — the LDG penalty policy for diffusive equations.

All components must be isbits (GPU-movable); `adapt` distributes over the
fields.
"""
struct DGPhysics{E <: AbstractEquation, NF, B <: Tuple, S, ST}
    equation            :: E
    numerical_flux      :: NF
    boundary_conditions :: B
    source              :: S
    stabilization       :: ST
end

function DGPhysics(equation::AbstractEquation;
                   boundary_conditions,
                   numerical_flux=default_numerical_flux(equation),
                   source=nothing,
                   stabilization=default_stabilization(equation))
    return DGPhysics(equation, numerical_flux, Tuple(boundary_conditions),
                     source, stabilization)
end

Adapt.@adapt_structure DGPhysics

nvariables(phys::DGPhysics) = nvariables(phys.equation)
has_diffusion(phys::DGPhysics) = has_diffusion(phys.equation)
