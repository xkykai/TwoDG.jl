module TwoDG

# Write your package code here.

export 
    Mesh,
    unique_rows,
    make_circle_mesh, make_square_mesh,
    fixmesh, mkt2f, setbndnbrs, createnodes,
    uniformlocalpnts,
    meshplot_curved

include("Master/Master.jl")
include("Utils/Utils.jl")
include("App/App.jl")
include("Meshes/Meshes.jl")

using .App
using .Master
using .Meshes
using .Utils

end
