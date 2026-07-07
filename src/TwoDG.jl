module TwoDG

export
    # meshes + reference element
    Mesh, ReferenceElement, Master,
    MeshGeometry, discretize, boundary_names,
    square_geometry, circle_geometry, lshape_geometry, box_geometry,
    unique_rows,
    make_circle_mesh, make_square_mesh, make_box_mesh,
    mkmesh_circle, make_circle_nodes, mkmesh_square, mkmesh_duct, mkmesh_trefftz, mkmesh_naca, mkmesh_lshape, mkmesh_box, mkmesh_distort!, gmsh_geometry,
    fixmesh, mkt2f, setbndnbrs, createnodes, uniref, cgmesh, mkf2f,
    norient, face_vertices, orientation_permutations, face_orientation,
    uniformlocalpnts, localpnts, localpnts1d, localpnts3d,
    get_local_face_nodes,
    meshplot, scaplot, meshplot_curved, save_vtk,
    gaussquad1d, gaussquad2d, gaussquad3d, newton_raphson,
    koornwinder1d, koornwinder2d, koornwinder3d,
    areacircle, trefftz_points, potential_trefftz, trefftz,
    # geometry cache
    GeometricFactors, SideGeometry, min_inscribed_diameter,
    # physics: equations, numerical fluxes, boundary conditions (the
    # extension contract — implement these methods on your own types)
    AbstractEquation,
    ConvectionEquation, ConvectionDiffusionEquation, WaveEquation,
    EulerEquations, PoissonEquation,
    nvariables, varnames, flux, normal_flux, max_abs_speed, has_diffusion,
    viscous_flux, viscous_numerical_flux, boundary_viscous_flux,
    BoundaryCondition, Dirichlet, Neumann, SlipWall, FarField, IncomingWave,
    boundary_flux, boundary_state, boundary_trace,
    RoeFlux, LaxFriedrichs, default_numerical_flux,
    LDGStabilization, default_stabilization,
    density, velocity, pressure, soundspeed, mach, entropy,
    energy_kinetic, energy_internal, energy_total, derived_field,
    wavespeed, diffusivity,
    eulereval, riemann_to_canonical, canonical_to_riemann,
    # CG
    elemmat_cg, cg_solve, cg_parsolve,
    grad_u, equilibrate, reconstruct,
    initu, interpolate, l2error,
    # DG (KernelAbstractions path)
    DGContext, DGPhysics, RinvWorkspace, rinvexpl!, rinvexpl_ka, rk4_ka!,
    RldgWorkspace, getq!, getq_ka, rldgexpl!, rldgexpl_ka,
    inviscid_residual!, viscous_residual!, compute_gradient!,
    # HDG
    localprob, elemmat_hdg, hdg_solve, hdg_postprocess, hdg_parsolve,
    match_geometry!,
    HDGSystem, hdg_gmres_ka, hdg_parsolve_ka,
    HDGBatch, hdg_local_solves, hdg_recover, hdg_parsolve_batched,
    hdg_direct_batched, hdg_trace_system,
    hdg_ns_step, hdg_ns_solve, hdg_cd_step, hdg_ns_postprocess,
    HDGNSBatch, HDGCDBatch, HDGNSCache, HDGCDCache,
    hdg_ns_step_batched, hdg_cd_step_batched,
    # callbacks & run-time diagnostics (Callbacks module)
    SolveState, CallbackSet,
    EveryStep, IterationInterval, TimeInterval, SpecifiedTimes, WallTimeInterval,
    ProgressCallback, AnalysisCallback, SteadyStateCallback,
    SaveSolutionCallback, CheckpointCallback, StepsizeCallback,
    NaNCheckCallback,
    integrate, l2norm,
    # high-level problem/solve API (Interface module)
    solve, semidiscretize, compute_dt,
    DGProblem, HDGProblem, CGProblem,
    RK4, Direct, GMRES, ConjugateGradient

include("Utils/Utils.jl")
include("Meshes/Meshes.jl")
include("Masters/Masters.jl")
include("Geometry/Geometry.jl")
include("Equations/Equations.jl")
include("Drivers/Drivers.jl")
include("Plotting/Plotting.jl")
include("ContinuousGalerkin/ContinuousGalerkin.jl")
include("DiscontinuousGalerkin/DiscontinuousGalerkin.jl")
include("HybridizableDiscontinuousGalerkin/HybridizableDiscontinuousGalerkin.jl")
include("Callbacks/Callbacks.jl")
include("Interface/Interface.jl")

using .Drivers
using .Masters
using .Meshes
using .Utils
using .Geometry
using .Equations
using .Plotting
using .ContinuousGalerkin
using .DiscontinuousGalerkin
using .HybridizableDiscontinuousGalerkin
using .Callbacks
using .Interface

end
