module Meshes

using TwoDG.Utils

# Write your package code here.

export
    Mesh,
    make_circle_mesh, make_square_mesh,
    fixmesh, mkt2f, setbndnbrs, createnodes, uniref,
    mkmesh_circle, mkmesh_square, mkmesh_duct, mkmesh_trefftz, mkmesh_naca

include("make_meshes.jl")
include("node_preprocessing.jl")
include("mesh_formulation.jl")
include("makemesh_circle.jl")
include("makemesh_square.jl")
include("makemesh_duct.jl")
include("makemesh_trefftz.jl")
include("makemesh_naca.jl")

end
