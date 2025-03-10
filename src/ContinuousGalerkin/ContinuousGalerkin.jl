module ContinuousGalerkin

export 
    elemmat_cg, cg_solve,
    grad_u, equilibrate, reconstruct,
    l2_error

include("elemmat_cg.jl")
include("cg_solve.jl")
include("cg_bounds.jl")
include("l2_error.jl")
end