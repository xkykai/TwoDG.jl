module HybridizableDiscontinuousGalerkin

export localprob, elemmat_hdg, hdg_solve, hdg_postprocess, hdg_parsolve
export hdg_ns_step, hdg_ns_solve, hdg_cd_step, hdg_ns_postprocess

include("hdg_solve.jl")
include("hdg_postprocess.jl")
include("hdg_parsolve.jl")
include("hdg_navierstokes.jl")

end