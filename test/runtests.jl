using TwoDG
using Test

@testset "TwoDG.jl" begin
    include("test_masters.jl")
    include("test_meshes.jl")
    include("test_dg.jl")
    include("test_ka.jl")
    include("test_cg.jl")
    include("test_hdg.jl")
    include("test_interface.jl")
end
