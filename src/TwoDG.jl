module TwoDG

export 
    Mesh, Master,
    unique_rows,
    make_circle_mesh, make_square_mesh,
    mkmesh_circle, mkmesh_square, mkmesh_duct, mkmesh_trefftz, mkmesh_naca, mkmesh_lshape,
    fixmesh, mkt2f, setbndnbrs, createnodes, uniref, cgmesh,
    uniformlocalpnts, localpnts, localpnts1d,
    get_local_face_nodes,
    meshplot, scaplot, meshplot_curved,
    gaussquad1d, gaussquad2d, newton_raphson,
    koornwinder1d, koornwinder2d,
    areacircle, trefftz_points, potential_trefftz, trefftz,
    elemmat_cg, cg_solve,
    grad_u, equilibrate, reconstruct,
    initu, l2_error,
    App, mkapp_convection,
    rk4, rk4!, rinvexpl

include("Utils/Utils.jl")
include("Meshes/Meshes.jl")
include("Masters/Masters.jl")
include("Drivers/Drivers.jl")
include("Plotting/Plotting.jl")
include("ContinuousGalerkin/ContinuousGalerkin.jl")
include("DiscontinuousGalerkin/DiscontinuousGalerkin.jl")
include("Apps/Apps.jl")

using .Drivers
using .Masters
using .Meshes
using .Utils
using .Plotting
using .ContinuousGalerkin
using .DiscontinuousGalerkin
using .Apps

end
