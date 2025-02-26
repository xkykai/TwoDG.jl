module TwoDG

# Write your package code here.

export 
    Mesh, Master,
    unique_rows,
    make_circle_mesh, make_square_mesh,
    mkmesh_circle, mkmesh_square, mkmesh_duct, mkmesh_trefftz, mkmesh_naca,
    fixmesh, mkt2f, setbndnbrs, createnodes,
    uniformlocalpnts,
    meshplot, scaplot, meshplot_curved,
    gaussquad1d, gaussquad2d, newton_raphson,
    koornwinder1d, koornwinder2d,
    areacircle, trefftz_points, potential_trefftz, trefftz

include("Utils/Utils.jl")
include("Meshes/Meshes.jl")
include("Masters/Masters.jl")
include("App/App.jl")
include("Plotting/Plotting.jl")

using .App
using .Masters
using .Meshes
using .Utils
using .Plotting

end
