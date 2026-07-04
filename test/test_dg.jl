# DG discretization: physics invariants (free-stream preservation, exactness of
# the LDG gradient) and interpolation accuracy, on straight and curved meshes.
# All residuals go through the KA path (`DGContext` + `DGPhysics`) — the
# single DG implementation.

using TwoDG
using Test
using LinearAlgebra
using StaticArrays

@testset "Discontinuous Galerkin" begin
    @testset "Euler free-stream preservation (curved mesh)" begin
        γ = 1.4
        uinf = [1.0, 0.3, 0.05, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2)]
        mesh = mkmesh_trefftz(6, 12, 3)
        master = Master(mesh)

        phys = DGPhysics(EulerEquations(γ=γ);
                         boundary_conditions=(FarField(uinf), FarField(uinf)))
        u = initu(mesh, 4, uinf)

        ctx = DGContext(master, mesh)
        r = rinvexpl_ka(ctx, phys, u, 0.0)
        @test norm(r) / norm(u) < 1e-10
    end

    @testset "LDG gradient is exact for linear fields" begin
        # all-Neumann boundaries: the LDG trace û = u⁻ is exact for u ∈ P_p,
        # so q = ∇u must be reproduced to roundoff
        for mesh in (mkmesh_square(5, 5, 2, 0, 1), mkmesh_trefftz(6, 12, 3))
            master = Master(mesh)
            nbnd = maximum(-mesh.f[mesh.f[:, 4] .< 0, 4])

            eq = ConvectionDiffusionEquation(x -> SVector(-x[2], x[1]), 1.0)
            phys = DGPhysics(eq;
                             boundary_conditions=ntuple(_ -> Neumann(), nbnd),
                             stabilization=LDGStabilization(1.0, 0.0))
            u = initu(mesh, 1, [(x, y) -> 2x + 3y - 1])

            q = getq_ka(DGContext(master, mesh), phys, u, 0.0)
            @test maximum(abs, q[:, 1, 1, :] .- 2) < 1e-8
            @test maximum(abs, q[:, 2, 1, :] .- 3) < 1e-8
        end
    end

    @testset "nodal interpolation converges at O(h^{p+1}) (p = $p)" for p in (1, 3)
        exact(x, y) = sin(π * x) * cos(π * y)
        errs = map((5, 9)) do n
            mesh = mkmesh_square(n, n, p, 0, 1)
            u = initu(mesh, 1, [exact])
            l2error(mesh, u[:, 1, :], exact)
        end
        rate = log2(errs[1] / errs[2])
        @test rate > p + 0.5
    end
end
