# The single physics representation: equations,
# numerical fluxes, and boundary conditions are small immutable structs, and
# all behavior is selected by multiple dispatch on them. The same objects are
# the *user extension surface* — a downstream script defines
#
#     struct MyEquation <: AbstractEquation ... end
#     TwoDG.flux(eq::MyEquation, u, x, t) = ...
#     TwoDG.nvariables(::MyEquation) = ...
#
# (plus, optionally, a numerical-flux callable and `boundary_state` methods)
# and every solver path accepts it. No integer physics codes, no callback
# slots, no package edits.
module Equations

using StaticArrays
using LinearAlgebra: norm
using Adapt

export AbstractEquation,
       ConvectionEquation, ConvectionDiffusionEquation, WaveEquation,
       EulerEquations, PoissonEquation,
       nvariables, varnames, flux, normal_flux, max_abs_speed, has_diffusion,
       viscous_flux, viscous_numerical_flux, boundary_viscous_flux,
       BoundaryCondition, Dirichlet, Neumann, SlipWall, FarField, IncomingWave,
       boundary_flux, boundary_state, boundary_trace,
       RoeFlux, LaxFriedrichs, default_numerical_flux,
       LDGStabilization, default_stabilization,
       density, velocity, pressure, soundspeed, mach, entropy,
       energy_kinetic, energy_internal, energy_total, derived_field,
       wavespeed, diffusivity,
       eulereval, riemann_to_canonical, canonical_to_riemann

include("interface.jl")
include("boundary_conditions.jl")
include("numerical_fluxes.jl")
include("convection.jl")
include("convection_diffusion.jl")
include("wave.jl")
include("euler.jl")

end # module Equations
