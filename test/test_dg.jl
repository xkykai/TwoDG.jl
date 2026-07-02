# DG discretization: physics invariants (free-stream preservation, exactness of
# the LDG gradient) and interpolation accuracy, on straight and curved meshes.
# All residuals go through the KA path — the only implementation since the
# legacy matrix-flux path was retired (roadmap A2.2); the deprecated
# `(master, mesh, app, u, t)` shims are exercised here too.

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

        app = mkapp_euler_pt(; gamma=γ, bcm=[1, 1], bcs=reshape(uinf, 1, 4))
        u = initu(mesh, app, uinf)

        ctx = DGContext(master, mesh)
        r = rinvexpl_ka(ctx, app, u, 0.0)
        @test norm(r) / norm(u) < 1e-10

        # deprecated legacy-signature shim routes through the same kernels
        r_shim = rinvexpl(master, mesh, app, u, 0.0)
        @test r_shim == r
    end

    @testset "LDG gradient is exact for linear fields" begin
        # all-Neumann boundaries: the LDG trace û = u⁻ is exact for u ∈ P_p,
        # so q = ∇u must be reproduced to roundoff
        for mesh in (mkmesh_square(5, 5, 2, 0, 1), mkmesh_trefftz(6, 12, 3))
            master = Master(mesh)
            nbnd = maximum(-mesh.f[mesh.f[:, 4] .< 0, 4])

            app = mkapp_convection_diffusion_pt(x -> SVector(-x[2], x[1]);
                                                kappa=1.0, c11=1.0,
                                                bcm=fill(2, nbnd), bcs=zeros(2, 1))
            u = initu(mesh, app, [(x, y) -> 2x + 3y - 1])

            q = getq_ka(DGContext(master, mesh), app, u, 0.0)
            @test maximum(abs, q[:, 1, 1, :] .- 2) < 1e-8
            @test maximum(abs, q[:, 2, 1, :] .- 3) < 1e-8

            q_shim = getq(master, mesh, app, u, 0.0)
            @test q_shim == q
        end
    end

    @testset "legacy matrix-flux path is fully retired" begin
        mesh = mkmesh_square(5, 5, 2, 0, 1)
        master = Master(mesh)
        # Dict-arg apps (the legacy convention) are rejected with guidance
        app_legacy = App(; nc=1, arg=Dict(:vf => [1.0, 0.0]),
                         bcm=[1, 1, 1, 1], bcs=zeros(1, 1))
        u = initu(mesh, app_legacy, [(x, y) -> x])
        @test_throws ArgumentError rinvexpl(master, mesh, app_legacy, u, 0.0)
        # legacy constructors error with migration instructions
        @test_throws ErrorException mkapp_convection()
        @test_throws ErrorException mkapp_wave()
        @test_throws ErrorException mkapp_euler()
        @test_throws ErrorException mkapp_convection_diffusion()
    end

    @testset "nodal interpolation converges at O(h^{p+1}) (p = $p)" for p in (1, 3)
        exact(x, y) = sin(π * x) * cos(π * y)
        app = App(; nc=1)
        errs = map((5, 9)) do n
            mesh = mkmesh_square(n, n, p, 0, 1)
            u = initu(mesh, app, [exact])
            l2error(mesh, u[:, 1, :], exact)
        end
        rate = log2(errs[1] / errs[2])
        @test rate > p + 0.5
    end
end
