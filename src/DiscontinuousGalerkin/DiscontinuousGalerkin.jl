module DiscontinuousGalerkin

export rk4, rk4!, rinvexpl, rldgexpl, getq
export DGContext, RinvWorkspace, rinvexpl!, rinvexpl_ka, rk4_ka!
export RldgWorkspace, getq!, getq_ka, rldgexpl!, rldgexpl_ka

include("rk4.jl")
include("dg_context.jl")
include("rinvexpl_ka.jl")
include("rldgexpl_ka.jl")
include("legacy_shims.jl")

end