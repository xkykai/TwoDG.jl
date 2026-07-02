module HybridizableDiscontinuousGalerkin

export localprob, elemmat_hdg, hdg_solve, hdg_postprocess, hdg_parsolve
export hdg_ns_step, hdg_ns_solve, hdg_cd_step, hdg_ns_postprocess
export HDGSystem, hdg_gmres_ka, hdg_parsolve_ka
export HDGBatch, hdg_local_solves, hdg_recover, hdg_parsolve_batched

include("hdg_solve.jl")
include("hdg_postprocess.jl")
include("hdg_parsolve.jl")
include("hdg_ka.jl")
include("hdg_batch.jl")
include("hdg_navierstokes.jl")

end