module TwoDG

export
    Mesh, Master,
    MeshGeometry, discretize, boundary_names,
    square_geometry, circle_geometry, lshape_geometry,
    unique_rows,
    make_circle_mesh, make_square_mesh,
    mkmesh_circle, make_circle_nodes, mkmesh_square, mkmesh_duct, mkmesh_trefftz, mkmesh_naca, mkmesh_lshape, mkmesh_distort!,
    fixmesh, mkt2f, setbndnbrs, createnodes, uniref, cgmesh, mkf2f,
    uniformlocalpnts, localpnts, localpnts1d,
    get_local_face_nodes,
    meshplot, scaplot, meshplot_curved,
    gaussquad1d, gaussquad2d, newton_raphson,
    koornwinder1d, koornwinder2d,
    areacircle, trefftz_points, potential_trefftz, trefftz,
    elemmat_cg, cg_solve, cg_parsolve,
    grad_u, equilibrate, reconstruct,
    initu, interpolate, l2error, l2_error,
    App, mkapp_convection, mkapp_wave, mkapp_euler, eulereval, mkapp_convection_diffusion,
    riemann_to_canonical, canonical_to_riemann,
    rk4, rk4!, rinvexpl, rldgexpl, getq,
    DGContext, RinvWorkspace, rinvexpl!, rinvexpl_ka, rk4_ka!,
    RldgWorkspace, getq!, getq_ka, rldgexpl!, rldgexpl_ka,
    mkapp_convection_pt, mkapp_wave_pt, mkapp_euler_pt,
    mkapp_convection_diffusion_pt,
    localprob, elemmat_hdg, hdg_solve, hdg_postprocess, hdg_parsolve,
    HDGSystem, hdg_gmres_ka, hdg_parsolve_ka,
    HDGBatch, hdg_local_solves, hdg_recover, hdg_parsolve_batched,
    hdg_ns_step, hdg_ns_solve, hdg_cd_step, hdg_ns_postprocess,
    # high-level problem/solve API (Interface module)
    solve, semidiscretize, compute_dt,
    ConvectionEquation, ConvectionDiffusionEquation, WaveEquation,
    EulerEquations, PoissonEquation, nvariables,
    Dirichlet, Neumann, SlipWall, FarField, IncomingWave,
    DGProblem, HDGProblem, CGProblem,
    RK4, Direct, GMRES, ConjugateGradient

include("Utils/Utils.jl")
include("Meshes/Meshes.jl")
include("Masters/Masters.jl")
include("Geometry/Geometry.jl")
include("Drivers/Drivers.jl")
include("Plotting/Plotting.jl")
include("ContinuousGalerkin/ContinuousGalerkin.jl")
include("DiscontinuousGalerkin/DiscontinuousGalerkin.jl")
include("Apps/Apps.jl")
include("HybridizableDiscontinuousGalerkin/HybridizableDiscontinuousGalerkin.jl")
include("Interface/Interface.jl")

using .Drivers
using .Masters
using .Meshes
using .Utils
using .Geometry
using .Plotting
using .ContinuousGalerkin
using .DiscontinuousGalerkin
using .HybridizableDiscontinuousGalerkin
using .Apps
using .Interface

end
