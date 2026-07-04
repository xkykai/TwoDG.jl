module ContinuousGalerkin

export
    elemmat_cg, cg_solve, cg_parsolve,
    cg_element_system, cg_assemble, CGMatVecOp,
    grad_u, equilibrate, reconstruct,
    l2error

include("elemmat_cg.jl")
include("cg_batch.jl")
include("cg_solve.jl")
include("cg_bounds.jl")
include("l2_error.jl")
end