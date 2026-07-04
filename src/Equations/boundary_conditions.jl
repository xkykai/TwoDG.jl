# Boundary conditions as user-extensible types. A boundary condition is a
# small immutable struct; its behavior is defined by dispatch on
# `boundary_state` (ghost state seen by the numerical flux), or — when a
# ghost-state formulation is not natural — by overriding `boundary_flux`
# directly. Viscous (LDG) problems additionally define `boundary_trace`
# (the trace û used by the gradient) and `boundary_viscous_flux`.
#
# GPU note: problems carry their BCs as a Tuple, one entry per boundary tag.
# Kernels select the entry by compile-time-unrolled recursion on the face's
# integer tag (`apply_boundary_flux` below), so the *dispatch* is static per
# tuple slot; the integer is data (which boundary), never code (which physics).

"""
    BoundaryCondition

Supertype of boundary conditions ([`Dirichlet`](@ref), [`Neumann`](@ref),
[`SlipWall`](@ref), [`FarField`](@ref), [`IncomingWave`](@ref), or a user
type). Implement, on your own type,

    boundary_state(bc::MyBC, eq, uL, n, x, t) -> SVector    # ghost state

(the numerical flux is then evaluated against it), or override

    boundary_flux(bc::MyBC, eq, numerical_flux, uL, n, x, t) -> SVector

for non-ghost-state conditions. Fields must be isbits (numbers, `SVector`s,
or closures over them) so the condition can move to a GPU.
"""
abstract type BoundaryCondition end

"""
    Dirichlet(value=0.0)

Prescribed solution value on a boundary: a constant, an `SVector` (one entry
per component), or a function `(x, t) -> value`.
"""
struct Dirichlet{G} <: BoundaryCondition
    value :: G
end
Dirichlet() = Dirichlet(0.0)

"""
    Neumann(flux=0.0)

Prescribed (currently homogeneous) normal flux: the solution trace is taken
from the interior, and the viscous boundary flux is `flux`.
"""
struct Neumann{G} <: BoundaryCondition
    flux :: G
end
Neumann() = Neumann(0.0)

"Impermeable slip wall (normal-velocity reflection) for the wave and Euler systems."
struct SlipWall <: BoundaryCondition end

"""
    FarField(state)

Far-field boundary carrying the free-stream `state` (one value per
component); the numerical flux against it upwinds automatically.
"""
struct FarField{S} <: BoundaryCondition
    state :: S
end
FarField(state::SVector) = FarField{typeof(state)}(state)
FarField(state::AbstractVector) = FarField(SVector{length(state)}(state...))

"Incoming-wave boundary for [`WaveEquation`](@ref) (uses the equation's `k`, `f`)."
struct IncomingWave <: BoundaryCondition end

# ------------------------------------------------------------- data adapters

# Evaluate boundary data against the interior state's shape/precision:
# numbers broadcast to all components, SVectors convert, callables get (x, t).
@inline bc_data(v::Number, uL::SVector{NC, T}, x, t) where {NC, T} =
    SVector{NC, T}(ntuple(_ -> convert(T, v), Val(NC)))
@inline bc_data(v::SVector{NC}, uL::SVector{NC, T}, x, t) where {NC, T} =
    SVector{NC, T}(v)
@inline function bc_data(g, uL::SVector{NC, T}, x, t) where {NC, T}
    val = g(x, t)
    return val isa Number ? SVector{NC, T}(ntuple(_ -> convert(T, val), Val(NC))) :
                            SVector{NC, T}(val)
end

# ------------------------------------------------------- the flux contract

"""
    boundary_state(bc, eq, uL, n, x, t) -> SVector

Ghost (exterior) state a boundary condition presents to the numerical flux:
the boundary data for [`Dirichlet`](@ref)/[`FarField`](@ref), the interior
state for [`Neumann`](@ref), the reflected state for [`SlipWall`](@ref), ….
"""
@inline boundary_state(bc::Dirichlet, eq, uL, n, x, t) = bc_data(bc.value, uL, x, t)
@inline boundary_state(bc::FarField, eq, uL, n, x, t) = bc_data(bc.state, uL, x, t)
@inline boundary_state(bc::Neumann, eq, uL, n, x, t) = uL

"""
    boundary_flux(bc, eq, numerical_flux, uL, n, x, t) -> SVector

Normal boundary flux: by default the numerical flux evaluated against the
condition's [`boundary_state`](@ref). Override for conditions that prescribe
the flux itself.
"""
@inline boundary_flux(bc::BoundaryCondition, eq, numerical_flux, uL, n, x, t) =
    numerical_flux(eq, uL, boundary_state(bc, eq, uL, n, x, t), n, x, t)

"""
    boundary_trace(bc, eq, uL, n, x, t) -> SVector

Solution trace û a boundary condition prescribes for the LDG gradient:
the boundary data for [`Dirichlet`](@ref), the interior trace for
[`Neumann`](@ref).
"""
@inline boundary_trace(bc::Dirichlet, eq, uL, n, x, t) = bc_data(bc.value, uL, x, t)
@inline boundary_trace(bc::Neumann, eq, uL, n, x, t) = uL

# ---------------------------------------------- tuple selection (GPU-static)

# Select the face's BC from the problem's Tuple by unrolled recursion on the
# boundary tag: each recursion level is compiled away, so the call below it
# dispatches statically — no dynamic dispatch on the device. An out-of-range
# tag falls through to the last entry; tags are validated at problem setup.
@inline apply_boundary_flux(bcs::Tuple{Any}, ib, eq, nf, uL, n, x, t) =
    boundary_flux(bcs[1], eq, nf, uL, n, x, t)
@inline apply_boundary_flux(bcs::Tuple, ib, eq, nf, uL, n, x, t) =
    ib == 1 ? boundary_flux(bcs[1], eq, nf, uL, n, x, t) :
              apply_boundary_flux(Base.tail(bcs), ib - 1, eq, nf, uL, n, x, t)

@inline apply_boundary_trace(bcs::Tuple{Any}, ib, eq, uL, n, x, t) =
    boundary_trace(bcs[1], eq, uL, n, x, t)
@inline apply_boundary_trace(bcs::Tuple, ib, eq, uL, n, x, t) =
    ib == 1 ? boundary_trace(bcs[1], eq, uL, n, x, t) :
              apply_boundary_trace(Base.tail(bcs), ib - 1, eq, uL, n, x, t)

@inline apply_boundary_viscous_flux(bcs::Tuple{Any}, ib, stab, eq, uL, qL, n, x, t) =
    boundary_viscous_flux(bcs[1], stab, eq, uL, qL, n, x, t)
@inline apply_boundary_viscous_flux(bcs::Tuple, ib, stab, eq, uL, qL, n, x, t) =
    ib == 1 ? boundary_viscous_flux(bcs[1], stab, eq, uL, qL, n, x, t) :
              apply_boundary_viscous_flux(Base.tail(bcs), ib - 1, stab, eq, uL, qL, n, x, t)
