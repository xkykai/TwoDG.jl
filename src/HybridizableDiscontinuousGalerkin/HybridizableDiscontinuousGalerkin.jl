module HybridizableDiscontinuousGalerkin

export localprob, elemmat_hdg, hdg_solve, hdg_postprocess, hdg_parsolve
export hdg_ns_step, hdg_ns_solve, hdg_cd_step, hdg_ns_postprocess
export HDGSystem, hdg_gmres_ka, hdg_parsolve_ka
export HDGBatch, hdg_local_solves, hdg_recover, hdg_parsolve_batched,
       hdg_direct_batched, hdg_trace_system
export HDGNSBatch, HDGCDBatch, HDGNSCache, HDGCDCache,
       hdg_ns_step_batched, hdg_cd_step_batched

include("hdg_solve.jl")          # per-element reference implementation (test oracle)
include("hdg_postprocess.jl")
include("hdg_parsolve.jl")       # threaded per-element assembly + GMRES reference
include("hdg_ka.jl")             # matrix-free trace system + KA GMRES
include("hdg_batch.jl")          # THE assembly engine: batched local solves,
                                 # Direct/GMRES as algorithm choices over it
include("hdg_navierstokes.jl")   # per-element NS reference (parity oracle)
# batched Navier-Stokes / scalar transport, split by concern:
include("hdg_ns_operators.jl")   # geometry × (ν, τ) constants, built once
include("hdg_ns_kernels.jl")     # Newton linearization + recovery KA kernels
include("hdg_ns_solve_batched.jl") # trace pattern, cache, NS driver
include("hdg_cd_batch.jl")       # scalar transport (Boussinesq temperature)

end