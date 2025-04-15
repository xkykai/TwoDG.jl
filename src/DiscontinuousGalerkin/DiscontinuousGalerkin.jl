module DiscontinuousGalerkin

export rk4, rk4!, rinvexpl, rldgexpl, getq

include("rk4.jl")
include("rinvexpl.jl")
include("rldgexpl.jl")

end