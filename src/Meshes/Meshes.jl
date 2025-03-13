module Meshes

using TwoDG.Utils

# Write your package code here.

export
    Mesh,
    make_circle_mesh, make_square_mesh,
    fixmesh, mkt2f, setbndnbrs, createnodes, uniref, cgmesh,
    mkmesh_circle, mkmesh_square, mkmesh_duct, mkmesh_trefftz, mkmesh_naca, mkmesh_lshape

include("make_meshes.jl")
include("node_preprocessing.jl")
include("mesh_formulation.jl")
include("cg_mesh.jl")
include("makemesh_circle.jl")
include("makemesh_square.jl")
include("makemesh_duct.jl")
include("makemesh_trefftz.jl")
include("makemesh_naca.jl")
include("makemesh_lshape.jl")

end
