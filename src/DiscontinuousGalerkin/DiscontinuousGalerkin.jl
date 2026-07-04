module DiscontinuousGalerkin

export DGContext, DGPhysics, RinvWorkspace, rinvexpl!, rinvexpl_ka, rk4_ka!
export RldgWorkspace, getq!, getq_ka, rldgexpl!, rldgexpl_ka
export inviscid_residual!, viscous_residual!, compute_gradient!

include("dg_context.jl")
include("dg_physics.jl")
include("rinvexpl_ka.jl")
include("rldgexpl_ka.jl")

# Readable primary names (REFACTOR_PLAN §6 naming migration). The terse
# MATLAB-derived names remain as aliases for one release; new code and docs
# use these.
"Inviscid DG residual — primary name for [`rinvexpl!`](@ref)."
const inviscid_residual! = rinvexpl!
"LDG viscous DG residual — primary name for [`rldgexpl!`](@ref)."
const viscous_residual! = rldgexpl!
"LDG gradient computation — primary name for [`getq!`](@ref)."
const compute_gradient! = getq!

end
