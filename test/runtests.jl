using TwoDG
using Test

@testset "TwoDG.jl" begin
    include("test_aqua.jl")
    include("test_masters.jl")
    include("test_meshes.jl")
    include("test_3d.jl")
    include("test_dg.jl")
    include("test_ka.jl")
    include("test_cg.jl")
    include("test_hdg.jl")
    include("test_interface.jl")
    include("test_vtk.jl")    # self-skips unless WriteVTK.jl is installed
    include("test_gmsh.jl")   # self-skips unless Gmsh.jl is installed
end
