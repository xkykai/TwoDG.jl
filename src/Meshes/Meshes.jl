module Meshes

using TwoDG.Utils

# Write your package code here.

export
    Mesh, MeshGeometry, discretize, boundary_names,
    square_geometry, circle_geometry, lshape_geometry, box_geometry,
    make_circle_mesh, make_square_mesh, make_box_mesh,
    fixmesh, mkt2f, setbndnbrs, createnodes, uniref, cgmesh, mkf2f,
    norient, face_vertices, orientation_permutations, face_orientation,
    mkmesh_circle, make_circle_nodes, mkmesh_square, mkmesh_duct, mkmesh_trefftz, mkmesh_naca, mkmesh_lshape,
    mkmesh_box, mkmesh_distort!, gmsh_geometry

include("orientation.jl")
include("make_meshes.jl")
include("node_preprocessing.jl")
include("mesh_formulation.jl")
include("geometry_stage.jl")
include("cg_mesh.jl")
include("makemesh_circle.jl")
include("makemesh_square.jl")
include("makemesh_box.jl")
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

"""
    gmsh_geometry(filepath; boundaries, curved=Symbol[], fd=nothing) -> MeshGeometry

Read a linear Gmsh `.msh` file into a [`MeshGeometry`](@ref): tetrahedra give
a `MeshGeometry{3}`, triangles a `MeshGeometry{2}`. Duplicate nodes are
merged and elements positively oriented ([`fixmesh`](@ref)); `boundaries`,
`curved`, `fd` as in [`MeshGeometry`](@ref) (curved boundaries are projected
onto their signed-distance zero sets by [`discretize`](@ref)). Implemented in
the `TwoDGGmshExt` package extension: available once Gmsh.jl is loaded
(`using TwoDG, Gmsh`).
"""
gmsh_geometry(args...; kwargs...) =
    error("gmsh_geometry requires Gmsh.jl. Load it first: `using Gmsh`.")

end
