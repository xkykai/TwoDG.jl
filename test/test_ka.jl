# The KernelAbstractions residual path (DGContext + pointwise fluxes) is the
# only DG implementation since the legacy matrix-flux path was retired
# (roadmap A2.2). Where the old suite checked parity against the legacy code,
# this one validates against exact/manufactured solutions and pins the
# (formerly parity-validated) flux implementations with golden regression
# values.

using TwoDG
using Test
using LinearAlgebra
using StaticArrays

relerr(a, b) = norm(a .- b) / norm(b)

# Golden regression values (see the "golden regression" testset below).
const GOLDEN_EULER_TREFFTZ = 0.25223786430401296
const GOLDEN_WAVE_SQUARE = 328.84364872457604

# smooth, strictly physical Euler state (γ = 1.4)
function euler_ic(γ)
    ρ(x, y) = 1.0 + 0.1 * exp(-30 * ((x - 0.5)^2 + (y - 0.5)^2))
    ρu(x, y) = 0.3 * ρ(x, y)
    ρv(x, y) = 0.05 * ρ(x, y)
    ρE(x, y) = 1.0 / (γ - 1) + 0.5 * (ρu(x, y)^2 + ρv(x, y)^2) / ρ(x, y)
    return [ρ, ρu, ρv, ρE]
end

@testset "KernelAbstractions residual path" begin
    @testset "convection: exact translation, O(h^{p+1}) convergence" begin
        # u_t + ∇·(v u) = 0 with constant v transports the profile unchanged:
        # u(x, t) = u0(x - v t). The bump stays far from the boundary, so the
        # far-field BC contributes only a ~1e-11 tail.
        v = SVector(1.0, 0.5)
        u0(x, y) = exp(-100 * ((x - 0.4)^2 + (y - 0.45)^2))
        dt, nstep = 5e-4, 200
        tfinal = dt * nstep
        exact(x, y) = u0(x - v[1] * tfinal, y - v[2] * tfinal)

        porder = 3
        errs = map((9, 13)) do n
            mesh = mkmesh_square(n, n, porder, 0, 1)
            master = Master(mesh)
            app = mkapp_convection_pt(v; bcm=fill(1, 4), bcs=zeros(1, 1))
            u = initu(mesh, app, [u0])
            rk4_ka!(DGContext(master, mesh), app, u, 0.0, dt, nstep)
            l2error(mesh, u[:, 1, :], exact)
        end
        @test errs[2] < 2e-3
        @test log(errs[1] / errs[2]) / log(12 / 8) > porder + 0.5
    end

    @testset "heat equation (LDG): exact decay, O(h^{p+1}) convergence" begin
        # u_t = κ Δu on the unit square with homogeneous Dirichlet boundaries:
        # u = sin(πx) sin(πy) exp(-2π²κt) is exact.
        κ = 0.05
        u0(x, y) = sin(π * x) * sin(π * y)
        dt, nstep = 2e-4, 500
        tfinal = dt * nstep
        exact(x, y) = u0(x, y) * exp(-2π^2 * κ * tfinal)

        porder = 2
        errs = map((5, 9)) do n
            mesh = mkmesh_square(n, n, porder, 0, 1)
            master = Master(mesh)
            app = mkapp_convection_diffusion_pt(SVector(0.0, 0.0);
                                                kappa=κ, c11=10.0, c11int=0.0,
                                                bcm=fill(1, 4), bcs=zeros(1, 1))
            u = initu(mesh, app, [u0])
            ctx = DGContext(master, mesh)
            rk4_ka!(rldgexpl!, ctx, app, u, 0.0, dt, nstep)
            l2error(mesh, u[:, 1, :], exact)
        end
        @test errs[2] < 5e-4
        @test log2(errs[1] / errs[2]) > porder + 0.5
    end

    @testset "golden regression: Euler and wave residuals" begin
        # Residual norms pinned 2026-07-02 from the implementation that was
        # validated to < 1e-9 relative error against the (now deleted) legacy
        # matrix-flux path on these exact meshes/states. Guards the flux
        # physics now that no independent implementation remains in-tree.
        γ = 1.4
        uinf = [1.0, 0.3, 0.05, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2)]
        mesh = mkmesh_trefftz(8, 16, 3)   # curved: exercises isoparametric geometry
        master = Master(mesh)

        app = mkapp_euler_pt(; gamma=γ, bcm=[1, 1], bcs=reshape(uinf, 1, 4))
        u = initu(mesh, app, euler_ic(γ))
        r = rinvexpl_ka(DGContext(master, mesh), app, u, 0.0)
        @test norm(r) ≈ GOLDEN_EULER_TREFFTZ rtol = 1e-9

        mesh = mkmesh_square(7, 7, 3, 0, 1)
        master = Master(mesh)
        c, k = 1.5, SVector(2.0, 1.0)
        fwave(c, k, x, t) = sin(k[1] * x[1] + k[2] * x[2] - c * hypot(k[1], k[2]) * t)
        app = mkapp_wave_pt(; c, k, f=fwave, bcm=[1, 2, 3, 2], bcs=zeros(3, 3))
        bump(x, y) = exp(-10 * ((x - 0.4)^2 + (y - 0.6)^2))
        u = initu(mesh, app, [bump, (x, y) -> 0.0, (x, y) -> 0.5 * bump(x, y)])
        r = rinvexpl_ka(DGContext(master, mesh), app, u, 0.3)
        @test norm(r) ≈ GOLDEN_WAVE_SQUARE rtol = 1e-9
    end

    @testset "legacy-signature shims drive rk4 unchanged (Euler)" begin
        γ = 1.4
        uinf = [1.0, 0.3, 0.05, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2)]
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        master = Master(mesh)
        app = mkapp_euler_pt(; gamma=γ, bcm=fill(1, 4), bcs=reshape(uinf, 1, 4))
        u0 = initu(mesh, app, euler_ic(γ))
        dt, nstep = 1e-3, 5

        u_shim = rk4(rinvexpl, master, mesh, app, copy(u0), 0.0, dt, nstep)
        u_ka = rk4_ka!(DGContext(master, mesh), app, copy(u0), 0.0, dt, nstep)
        @test relerr(u_shim, u_ka) < 1e-13
    end

    @testset "LDG viscous path: rk4 shim + Float32 (convection-diffusion)" begin
        κ, c11, c11int = 0.01, 10.0, 0.5
        make_pt(bcm, bcs) =
            mkapp_convection_diffusion_pt(x -> SVector(-x[2], x[1]);
                                          kappa=κ, c11, c11int, bcm, bcs)

        mesh = mkmesh_square(9, 9, 3, 0, 1)
        master = Master(mesh)
        bcm, bcs = [1, 2, 1, 2], zeros(2, 1)
        app = make_pt(bcm, bcs)
        u0 = initu(mesh, app, [(x, y) -> exp(-4 * ((x - 0.5)^2 + y^2))])
        ctx = DGContext(master, mesh)

        dt, nstep = 1e-4, 5
        u_shim = rk4(rldgexpl, master, mesh, app, copy(u0), 0.0, dt, nstep)
        u_ka = rk4_ka!(rldgexpl!, ctx, app, copy(u0), 0.0, dt, nstep)
        @test relerr(u_shim, u_ka) < 1e-13

        # curved mesh: viscous residual finite and Float32-consistent
        mesh_c = mkmesh_trefftz(8, 16, 3)
        master_c = Master(mesh_c)
        app_c = make_pt([1, 2], zeros(2, 1))
        u_c = initu(mesh_c, app_c, [(x, y) -> exp(-4 * ((x - 0.5)^2 + y^2))])
        ctx_c = DGContext(master_c, mesh_c)
        r_c = rldgexpl_ka(ctx_c, app_c, u_c, 0.11)
        @test all(isfinite, r_c)

        ctx32 = DGContext(master_c, mesh_c; T=Float32)
        r32 = rldgexpl_ka(ctx32, app_c, Float32.(u_c), 0.11f0)
        @test eltype(r32) == Float32
        @test all(isfinite, r32)
        @test relerr(Float64.(r32), r_c) < 1e-3
    end

    @testset "Float32 context (Euler)" begin
        γ = 1.4
        uinf = [1.0, 0.3, 0.05, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2)]
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        master = Master(mesh)
        app = mkapp_euler_pt(; gamma=γ, bcm=fill(1, 4), bcs=reshape(uinf, 1, 4))
        u = initu(mesh, app, euler_ic(γ))

        ctx64 = DGContext(master, mesh)
        r64 = rinvexpl_ka(ctx64, app, u, 0.0)

        ctx32 = DGContext(master, mesh; T=Float32)
        @test eltype(ctx32) == Float32
        r32 = rinvexpl_ka(ctx32, app, Float32.(u), 0.0f0)
        @test eltype(r32) == Float32
        @test all(isfinite, r32)
        @test relerr(Float64.(r32), r64) < 1e-3
    end
end
