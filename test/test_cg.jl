# Continuous Galerkin: Poisson with a manufactured solution, p-convergence.

using TwoDG
using Test

@testset "Continuous Galerkin Poisson" begin
    exact(x, y) = sin(π * x) * sin(π * y)
    source(x, y) = 2π^2 * exact(x, y)
    param = (; κ=1.0, c=[0.0, 0.0], s=0.0)

    errs = map((1, 2, 3)) do porder
        mesh = mkmesh_square(9, 9, porder, 0, 1)
        master = Master(mesh, 4porder)
        uh, energy = cg_solve(mesh, master, source, param)
        sqrt(l2_error(mesh, uh, exact))
    end

    @test errs[1] < 3e-2
    @test all(errs[i + 1] < errs[i] / 4 for i in 1:2)  # rapid p-convergence
end
