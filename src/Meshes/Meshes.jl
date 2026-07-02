module Meshes

using TwoDG.Utils

# Write your package code here.

export
    Mesh,
    make_circle_mesh, make_square_mesh,
    fixmesh, mkt2f, setbndnbrs, createnodes, uniref, cgmesh, mkf2f,
    mkmesh_circle, make_circle_nodes, mkmesh_square, mkmesh_duct, mkmesh_trefftz, mkmesh_naca, mkmesh_lshape,
    mkmesh_distort!

include("make_meshes.jl")
include("node_preprocessing.jl")
include("mesh_formulation.jl")
include("cg_mesh.jl")
include("makemesh_circle.jl")
include("makemesh_square.jl")
include("makemesh_duct.jl")
include("makemesh_trefftz.jl")
include("makemesh_lshape.jl")

"""
    mkmesh_naca(t_naca=10, porder=2, name="naca0012", display_gmsh=false)

Generate a curved high-order mesh around a NACA 4-digit symmetric airfoil
(thickness `t_naca` in percent of chord). Implemented in the `TwoDGGmshExt`
package extension: it becomes available once Gmsh.jl is loaded
(`using TwoDG, Gmsh`).
"""
mkmesh_naca(args...; kwargs...) =
    error("mkmesh_naca requires Gmsh.jl. Load it first: `using Gmsh`.")

end
