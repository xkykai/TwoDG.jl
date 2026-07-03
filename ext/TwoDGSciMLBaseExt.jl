# Package extension: `semidiscretize(prob::DGProblem, tspan) -> ODEProblem`
# bridge to the SciML ecosystem. Loaded automatically when both TwoDG and
# SciMLBase are in the environment (any OrdinaryDiffEq solver package pulls
# SciMLBase in).
module TwoDGSciMLBaseExt

using TwoDG
using TwoDG.Interface: DGProblem, _dg_setup
using TwoDG.DiscontinuousGalerkin: rinvexpl!, rldgexpl!, RinvWorkspace, RldgWorkspace
import SciMLBase

# The DG residual is already M⁻¹-applied, so it is du/dt directly. The
# workspace is preallocated once and closed over via the ODE parameters;
# kernels never allocate inside the RHS.
function _semidiscretize(prob::DGProblem, tspan; ArrayT=Array, ngauss=nothing)
    ctx, app, u0, residual! = _dg_setup(prob; ArrayT, ngauss)
    ws = residual! === rldgexpl! ? RldgWorkspace(ctx, app.nc) : RinvWorkspace(ctx, app.nc)
    params = (; ctx, app, ws, residual!)

    function dg_rhs!(du, u, p, t)
        p.residual!(du, p.ctx, p.app, u, t; ws=p.ws)
        return nothing
    end

    T = eltype(u0)
    return SciMLBase.ODEProblem(dg_rhs!, u0, (T(tspan[1]), T(tspan[2])), params)
end

end # module
