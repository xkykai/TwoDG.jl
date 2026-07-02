# Parity tests: the KernelAbstractions residual path (DGContext + pointwise
# fluxes + rinvexpl!) must reproduce the legacy threaded rinvexpl to roundoff
# on the KA CPU backend, for straight and curved meshes.

using TwoDG
using Test
using LinearAlgebra
using StaticArrays

relerr(a, b) = norm(a .- b) / norm(b)

# smooth, strictly physical Euler state (γ = 1.4)
function euler_ic(γ)
    ρ(x, y) = 1.0 + 0.1 * exp(-30 * ((x - 0.5)^2 + (y - 0.5)^2))
    ρu(x, y) = 0.3 * ρ(x, y)
    ρv(x, y) = 0.05 * ρ(x, y)
    ρE(x, y) = 1.0 / (γ - 1) + 0.5 * (ρu(x, y)^2 + ρv(x, y)^2) / ρ(x, y)
    return [ρ, ρu, ρv, ρE]
end

@testset "KernelAbstractions residual path" begin
    @testset "convection, square mesh (constant velocity + source)" begin
        porder = 3
        mesh = mkmesh_square(9, 9, porder, 0, 1)
        master = Master(mesh)
        nbnd = 4  # square has 4 boundaries

        # legacy app (matrix fluxes, Dict params)
        app = mkapp_convection()
        app.arg[:vf] = [1.0, 2.0]
        src_mat(ug, q, pg, arg, t) = reshape(sin.(pg[:, 1] .+ pg[:, 2]), :, 1)
        app = App(app; bcm=fill(1, nbnd), bcs=reshape([0.0], 1, 1), src=src_mat)

        # pointwise app
        src_pt(u, x, param, t) = SVector(sin(x[1] + x[2]))
        app_pt = mkapp_convection_pt(SVector(1.0, 2.0);
                                     bcm=fill(1, nbnd), bcs=reshape([0.0], 1, 1),
                                     src=src_pt)

        u = initu(mesh, app, [(x, y) -> exp(sin(3x) * cos(2y))])
        t = 0.37

        r_legacy = rinvexpl(master, mesh, app, u, t)
        ctx = DGContext(master, mesh)
        r_ka = rinvexpl_ka(ctx, app_pt, u, t)
        @test relerr(r_ka, r_legacy) < 1e-9
    end

    @testset "convection, Trefftz mesh (curved, velocity-field function)" begin
        porder = 3
        mesh = mkmesh_trefftz(8, 16, porder)  # pure-Julia curved mesh, 2 boundaries
        master = Master(mesh)
        @test any(mesh.fcurved) && any(mesh.tcurved)  # actually exercises curved paths

        app = mkapp_convection()
        app.arg[:vf] = p -> hcat(-p[:, 2], p[:, 1])  # rigid rotation, legacy matrix form
        app = App(app; bcm=[1, 1], bcs=reshape([0.0], 1, 1))

        app_pt = mkapp_convection_pt(x -> SVector(-x[2], x[1]);
                                     bcm=[1, 1], bcs=reshape([0.0], 1, 1))

        u = initu(mesh, app, [(x, y) -> exp(-4 * (x^2 + y^2))])
        t = 0.0

        r_legacy = rinvexpl(master, mesh, app, u, t)
        ctx = DGContext(master, mesh)
        r_ka = rinvexpl_ka(ctx, app_pt, u, t)
        @test relerr(r_ka, r_legacy) < 1e-9
    end

    @testset "Euler, square + Trefftz meshes" begin
        γ = 1.4
        uinf = [1.0, 0.3, 0.05, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2)]

        for (mesh, nbnd) in ((mkmesh_square(7, 7, 2, 0, 1), 4),
                             (mkmesh_trefftz(8, 16, 3), 2))
            master = Master(mesh)

            app = mkapp_euler()
            app.arg[:gamma] = γ
            app = App(app; bcm=fill(1, nbnd), bcs=reshape(uinf, 1, 4))

            app_pt = mkapp_euler_pt(; gamma=γ, bcm=fill(1, nbnd),
                                    bcs=reshape(uinf, 1, 4))

            u = initu(mesh, app, euler_ic(γ))
            t = 0.0

            r_legacy = rinvexpl(master, mesh, app, u, t)
            ctx = DGContext(master, mesh)
            r_ka = rinvexpl_ka(ctx, app_pt, u, t)
            @test relerr(r_ka, r_legacy) < 1e-9
        end
    end

    @testset "wave, square mesh (far-field/reflect/incoming boundaries)" begin
        mesh = mkmesh_square(7, 7, 3, 0, 1)
        master = Master(mesh)
        c, k = 1.5, [2.0, 1.0]
        fwave_leg(c, k, p, t) = sin.(k[1] .* p[:, 1] .+ k[2] .* p[:, 2] .- c * hypot(k[1], k[2]) * t)
        fwave_pt(c, k, x, t) = sin(k[1] * x[1] + k[2] * x[2] - c * hypot(k[1], k[2]) * t)

        bcm, bcs = [1, 2, 3, 2], zeros(3, 3)
        # legacy wave app ships with pg=false, but the incoming-wave BC needs
        # face coordinates — rebuild it with pg=true (the KA path always has them)
        app0 = mkapp_wave()
        app = App(; nc=app0.nc, pg=true, arg=app0.arg, bcm, bcs,
                  finvi=app0.finvi, finvb=app0.finvb, finvv=app0.finvv)
        app.arg[:c] = c
        app.arg[:k] = k
        app.arg[:f] = fwave_leg
        app_pt = mkapp_wave_pt(; c, k=SVector(k...), f=fwave_pt, bcm, bcs)

        bump(x, y) = exp(-10 * ((x - 0.4)^2 + (y - 0.6)^2))
        u = initu(mesh, app, [bump, (x, y) -> 0.0, (x, y) -> 0.5 * bump(x, y)])
        t = 0.3

        r_legacy = rinvexpl(master, mesh, app, u, t)
        r_ka = rinvexpl_ka(DGContext(master, mesh), app_pt, u, t)
        @test relerr(r_ka, r_legacy) < 1e-9
    end

    @testset "rk4 time stepping parity (Euler, square)" begin
        γ = 1.4
        uinf = [1.0, 0.3, 0.05, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2)]
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        master = Master(mesh)

        app = mkapp_euler()
        app.arg[:gamma] = γ
        app = App(app; bcm=fill(1, 4), bcs=reshape(uinf, 1, 4))
        app_pt = mkapp_euler_pt(; gamma=γ, bcm=fill(1, 4), bcs=reshape(uinf, 1, 4))

        u0 = initu(mesh, app, euler_ic(γ))
        dt, nstep = 1e-3, 5

        u_legacy = rk4(rinvexpl, master, mesh, app, copy(u0), 0.0, dt, nstep)
        ctx = DGContext(master, mesh)
        u_ka = rk4_ka!(ctx, app_pt, copy(u0), 0.0, dt, nstep)
        @test relerr(u_ka, u_legacy) < 1e-9
    end

    @testset "LDG viscous path (convection-diffusion)" begin
        κ, c11, c11int = 0.01, 10.0, 0.5

        # legacy app: matrix fluxes, Dict params, matrix velocity field
        function make_legacy(bcm, bcs)
            app = mkapp_convection_diffusion()
            app = App(app; bcm, bcs)
            app.arg[:vf] = p -> hcat(-p[:, 2], p[:, 1])  # rigid rotation
            app.arg[:kappa] = κ
            app.arg[:c11] = c11
            app.arg[:c11int] = c11int
            return app
        end
        make_pt(bcm, bcs) =
            mkapp_convection_diffusion_pt(x -> SVector(-x[2], x[1]);
                                          kappa=κ, c11, c11int, bcm, bcs)

        # mixed Dirichlet/Neumann boundaries on both a straight and a curved mesh
        for (mesh, bcm) in ((mkmesh_square(9, 9, 3, 0, 1), [1, 2, 1, 2]),
                            (mkmesh_trefftz(8, 16, 3), [1, 2]))
            master = Master(mesh)
            bcs = zeros(2, 1)
            app = make_legacy(bcm, bcs)
            app_pt = make_pt(bcm, bcs)

            u = initu(mesh, app, [(x, y) -> exp(-4 * ((x - 0.5)^2 + y^2))])
            t = 0.11
            ctx = DGContext(master, mesh)

            # LDG gradient parity
            q_legacy = getq(master, mesh, app, u, t)
            q_ka = getq_ka(ctx, app_pt, u, t)
            @test relerr(q_ka, q_legacy) < 1e-9

            # viscous residual parity
            r_legacy = rldgexpl(master, mesh, app, u, t)
            r_ka = rldgexpl_ka(ctx, app_pt, u, t)
            @test relerr(r_ka, r_legacy) < 1e-9
        end

        # time stepping parity + Float32 sanity (square mesh)
        mesh = mkmesh_square(9, 9, 3, 0, 1)
        master = Master(mesh)
        bcm, bcs = [1, 2, 1, 2], zeros(2, 1)
        app = make_legacy(bcm, bcs)
        app_pt = make_pt(bcm, bcs)
        u0 = initu(mesh, app, [(x, y) -> exp(-4 * ((x - 0.5)^2 + y^2))])
        ctx = DGContext(master, mesh)

        dt, nstep = 1e-4, 5
        u_legacy = rk4(rldgexpl, master, mesh, app, copy(u0), 0.0, dt, nstep)
        u_ka = rk4_ka!(rldgexpl!, ctx, app_pt, copy(u0), 0.0, dt, nstep)
        @test relerr(u_ka, u_legacy) < 1e-9

        ctx32 = DGContext(master, mesh; T=Float32)
        r32 = rldgexpl_ka(ctx32, app_pt, Float32.(u0), 0.0f0)
        @test eltype(r32) == Float32
        @test all(isfinite, r32)
        @test relerr(Float64.(r32), rldgexpl_ka(ctx, app_pt, u0, 0.0)) < 1e-3
    end

    @testset "Float32 context" begin
        γ = 1.4
        uinf = [1.0, 0.3, 0.05, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2)]
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        master = Master(mesh)
        app_pt = mkapp_euler_pt(; gamma=γ, bcm=fill(1, 4), bcs=reshape(uinf, 1, 4))

        u = initu(mesh, App(mkapp_euler(); bcm=fill(1, 4), bcs=reshape(uinf, 1, 4)),
                  euler_ic(γ))

        ctx64 = DGContext(master, mesh)
        r64 = rinvexpl_ka(ctx64, app_pt, u, 0.0)

        ctx32 = DGContext(master, mesh; T=Float32)
        @test eltype(ctx32) == Float32
        r32 = rinvexpl_ka(ctx32, app_pt, Float32.(u), 0.0f0)
        @test eltype(r32) == Float32
        @test all(isfinite, r32)
        @test relerr(Float64.(r32), r64) < 1e-3
    end
end
