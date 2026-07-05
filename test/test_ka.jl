# The KernelAbstractions residual path (DGContext + DGPhysics) is the only DG
# implementation. Where the old suite checked parity against the legacy code,
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

# --- user-defined physics for the extension-contract testset (top level:
# structs cannot be defined inside a @testset block). Burgers' equation, a
# frozen-ghost-state BC, and a central numerical flux, built from exported
# methods only.
struct BurgersEquation <: TwoDG.AbstractEquation{2} end
TwoDG.nvariables(::BurgersEquation) = 1
TwoDG.flux(::BurgersEquation, u::SVector{1, T}, x, t) where {T} =
    (u .* u ./ 2, u .* u ./ 2)
TwoDG.max_abs_speed(::BurgersEquation, u::SVector{1}, n, x, t) =
    abs(u[1] * (n[1] + n[2]))

struct FrozenBC <: TwoDG.BoundaryCondition end
TwoDG.boundary_state(::FrozenBC, eq, uL, n, x, t) = one.(uL)

struct CentralFlux end
(::CentralFlux)(eq, uL, uR, n, x, t) =
    (TwoDG.normal_flux(eq, uL, n, x, t) + TwoDG.normal_flux(eq, uR, n, x, t)) / 2

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
            phys = DGPhysics(ConvectionEquation(v);
                             boundary_conditions=ntuple(_ -> FarField(SVector(0.0)), 4))
            u = initu(mesh, 1, [u0])
            rk4_ka!(DGContext(master, mesh), phys, u, 0.0, dt, nstep)
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
            phys = DGPhysics(ConvectionDiffusionEquation(SVector(0.0, 0.0), κ);
                             boundary_conditions=ntuple(_ -> Dirichlet(0.0), 4),
                             stabilization=LDGStabilization(10.0, 0.0))
            u = initu(mesh, 1, [u0])
            ctx = DGContext(master, mesh)
            rk4_ka!(rldgexpl!, ctx, phys, u, 0.0, dt, nstep)
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

        phys = DGPhysics(EulerEquations(γ=γ);
                         boundary_conditions=(FarField(uinf), FarField(uinf)))
        u = initu(mesh, 4, euler_ic(γ))
        r = rinvexpl_ka(DGContext(master, mesh), phys, u, 0.0)
        @test norm(r) ≈ GOLDEN_EULER_TREFFTZ rtol = 1e-9

        mesh = mkmesh_square(7, 7, 3, 0, 1)
        master = Master(mesh)
        c, k = 1.5, SVector(2.0, 1.0)
        fwave(c, k, x, t) = sin(k[1] * x[1] + k[2] * x[2] - c * hypot(k[1], k[2]) * t)
        phys = DGPhysics(WaveEquation(c; k, f=fwave);
                         boundary_conditions=(FarField(SVector(0.0, 0.0, 0.0)),
                                              SlipWall(), IncomingWave(), SlipWall()))
        bump(x, y) = exp(-10 * ((x - 0.4)^2 + (y - 0.6)^2))
        u = initu(mesh, 3, [bump, (x, y) -> 0.0, (x, y) -> 0.5 * bump(x, y)])
        r = rinvexpl_ka(DGContext(master, mesh), phys, u, 0.3)
        @test norm(r) ≈ GOLDEN_WAVE_SQUARE rtol = 1e-9
    end

    @testset "user-defined equation, flux, and BC (extension contract)" begin
        # Burgers' equation defined *outside* the package using only exported
        # methods (types at the top of this file) — the acid test that the
        # physics surface is open. A constant state must be preserved through
        # the full residual path.
        mesh = mkmesh_square(5, 5, 2, 0, 1)
        phys = DGPhysics(BurgersEquation();
                         boundary_conditions=ntuple(_ -> FrozenBC(), 4),
                         numerical_flux=LaxFriedrichs())
        u = initu(mesh, 1, [1.0])
        r = rinvexpl_ka(DGContext(Master(mesh), mesh), phys, u, 0.0)
        @test norm(r) / norm(u) < 1e-10          # free-stream preservation

        phys_c = DGPhysics(BurgersEquation();
                           boundary_conditions=ntuple(_ -> FrozenBC(), 4),
                           numerical_flux=CentralFlux())
        r_c = rinvexpl_ka(DGContext(Master(mesh), mesh), phys_c, u, 0.0)
        @test norm(r_c) / norm(u) < 1e-10
    end

    @testset "LDG viscous path: curved mesh + Float32 (convection-diffusion)" begin
        κ, c11, c11int = 0.01, 10.0, 0.5
        eq = ConvectionDiffusionEquation(x -> SVector(-x[2], x[1]), κ)
        stab = LDGStabilization(c11, c11int)

        # curved mesh: viscous residual finite and Float32-consistent
        mesh_c = mkmesh_trefftz(8, 16, 3)
        master_c = Master(mesh_c)
        phys_c = DGPhysics(eq; boundary_conditions=(Dirichlet(0.0), Neumann()),
                           stabilization=stab)
        u_c = initu(mesh_c, 1, [(x, y) -> exp(-4 * ((x - 0.5)^2 + y^2))])
        ctx_c = DGContext(master_c, mesh_c)
        r_c = rldgexpl_ka(ctx_c, phys_c, u_c, 0.11)
        @test all(isfinite, r_c)

        ctx32 = DGContext(master_c, mesh_c; T=Float32)
        r32 = rldgexpl_ka(ctx32, phys_c, Float32.(u_c), 0.11f0)
        @test eltype(r32) == Float32
        @test all(isfinite, r32)
        @test relerr(Float64.(r32), r_c) < 1e-3
    end

    @testset "Float32 context (Euler)" begin
        γ = 1.4
        uinf = [1.0, 0.3, 0.05, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2)]
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        master = Master(mesh)
        phys = DGPhysics(EulerEquations(γ=γ);
                         boundary_conditions=ntuple(_ -> FarField(uinf), 4))
        u = initu(mesh, 4, euler_ic(γ))

        ctx64 = DGContext(master, mesh)
        r64 = rinvexpl_ka(ctx64, phys, u, 0.0)

        ctx32 = DGContext(master, mesh; T=Float32)
        @test eltype(ctx32) == Float32
        r32 = rinvexpl_ka(ctx32, phys, Float32.(u), 0.0f0)
        @test eltype(r32) == Float32
        @test all(isfinite, r32)
        @test relerr(Float64.(r32), r64) < 1e-3
    end
end
