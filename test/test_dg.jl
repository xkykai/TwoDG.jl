# DG discretization: physics invariants (free-stream preservation, exactness of
# the LDG gradient) and interpolation accuracy, on straight and curved meshes.

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

        app = mkapp_euler()
        app.arg[:gamma] = γ
        app = App(app; bcm=[1, 1], bcs=reshape(uinf, 1, 4))
        u = initu(mesh, app, uinf)

        r = rinvexpl(master, mesh, app, u, 0.0)
        @test norm(r) / norm(u) < 1e-10

        app_pt = mkapp_euler_pt(; gamma=γ, bcm=[1, 1], bcs=reshape(uinf, 1, 4))
        ctx = DGContext(master, mesh)
        r_ka = rinvexpl_ka(ctx, app_pt, u, 0.0)
        @test norm(r_ka) / norm(u) < 1e-10
    end

    @testset "LDG gradient is exact for linear fields" begin
        # all-Neumann boundaries: the LDG trace û = u⁻ is exact for u ∈ P_p,
        # so q = ∇u must be reproduced to roundoff
        for mesh in (mkmesh_square(5, 5, 2, 0, 1), mkmesh_trefftz(6, 12, 3))
            master = Master(mesh)
            nbnd = maximum(-mesh.f[mesh.f[:, 4] .< 0, 4])

            app = mkapp_convection_diffusion()
            app = App(app; bcm=fill(2, nbnd), bcs=zeros(2, 1))
            app.arg[:vf] = p -> hcat(-p[:, 2], p[:, 1])
            app.arg[:kappa] = 1.0
            app.arg[:c11] = 1.0
            app.arg[:c11int] = 0.0

            u = initu(mesh, app, [(x, y) -> 2x + 3y - 1])
            q = getq(master, mesh, app, u, 0.0)
            @test maximum(abs, q[:, 1, 1, :] .- 2) < 1e-8
            @test maximum(abs, q[:, 2, 1, :] .- 3) < 1e-8

            app_pt = mkapp_convection_diffusion_pt(x -> SVector(-x[2], x[1]);
                                                   kappa=1.0, c11=1.0,
                                                   bcm=fill(2, nbnd), bcs=zeros(2, 1))
            q_ka = getq_ka(DGContext(master, mesh), app_pt, u, 0.0)
            @test maximum(abs, q_ka[:, 1, 1, :] .- 2) < 1e-8
            @test maximum(abs, q_ka[:, 2, 1, :] .- 3) < 1e-8
        end
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
