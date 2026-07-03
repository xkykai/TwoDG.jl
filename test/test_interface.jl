# High-level problem/solve API: equations-as-structs, boundary-condition
# objects, and the CommonSolve `solve` entry point must reproduce the
# low-level drivers they wrap.

using TwoDG
using TwoDG.Interface: lower_bcs, lower_dbc
using Test
using LinearAlgebra
using StaticArrays
using SciMLBase: ODEProblem
using OrdinaryDiffEqTsit5: Tsit5

@testset "Interface (problems + solve)" begin
    @testset "boundary-condition lowering" begin
        eq = EulerEquations(γ=1.4)
        uinf = [1.0, 0.3, 0.05, 2.0]
        bcm, bcs = lower_bcs(eq, [FarField(uinf), SlipWall(), FarField(uinf)])
        @test bcm == [1, 2, 1]
        @test bcs[1, :] == uinf
        @test all(bcs[2, :] .== 0)

        # conflicting data for the same code must be rejected
        @test_throws ArgumentError lower_bcs(eq, [FarField(uinf), FarField(2 .* uinf)])
        # unsupported combination must be rejected
        @test_throws ArgumentError lower_bcs(eq, [Neumann()])

        cd = ConvectionDiffusionEquation([1.0, 0.5], 0.1)
        bcm, bcs = lower_bcs(cd, [Dirichlet(0.0), Neumann(), Dirichlet(0.0), Neumann()])
        @test bcm == [1, 2, 1, 2]

        dbc = lower_dbc(Dirichlet((x, y) -> x + 2y))
        @test dbc([1.0 2.0; 3.0 4.0]) ≈ [5.0, 11.0]
        @test lower_dbc(Dirichlet(0.5))([1.0 2.0]) == [0.5]
    end

    @testset "DGProblem/RK4 reproduces rk4_ka! (Euler)" begin
        γ = 1.4
        uinf = [1.0, 0.3, 0.05, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2)]
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        eq = EulerEquations(γ=γ)
        u0funcs = [(x, y) -> uinf[c] for c in 1:4]

        prob = DGProblem(eq, mesh; bc=fill(FarField(uinf), 4), u0=u0funcs)
        sol = solve(prob, RK4(); dt=1e-3, nstep=5)
        @test sol.t ≈ 5e-3

        # free-stream preservation through the whole high-level path
        uinf_arr = initu(mesh, mkapp_euler_pt(; gamma=γ, bcm=fill(1, 4),
                                              bcs=reshape(uinf, 1, 4)), u0funcs)
        @test norm(sol.u .- uinf_arr) / norm(uinf_arr) < 1e-10

        # identical to driving the KA stepper by hand
        master = Master(mesh)
        app_pt = mkapp_euler_pt(; gamma=γ, bcm=fill(1, 4), bcs=reshape(uinf, 1, 4))
        u_ref = rk4_ka!(DGContext(master, mesh), app_pt, copy(uinf_arr), 0.0, 1e-3, 5)
        @test sol.u ≈ u_ref rtol = 1e-12
    end

    @testset "DGProblem LDG (convection-diffusion) runs and decays" begin
        eq = ConvectionDiffusionEquation(x -> SVector(-x[2], x[1]), 0.01;
                                         c11=10.0, c11int=0.5)
        mesh = mkmesh_square(9, 9, 3, 0, 1)
        u0 = [(x, y) -> exp(-4 * ((x - 0.5)^2 + y^2))]
        prob = DGProblem(eq, mesh; bc=[Dirichlet(), Neumann(), Dirichlet(), Neumann()], u0)
        sol = solve(prob, RK4(); dt=1e-4, nstep=20)
        @test all(isfinite, sol.u)
        @test size(sol.u) == (size(mesh.dgnodes, 1), 1, size(mesh.dgnodes, 3))
    end

    @testset "HDGProblem: Direct vs GMRES vs low-level" begin
        exact(x, y) = sin(π * x) * sin(π * y)
        source(p) = reshape(2π^2 .* sin.(π .* p[:, 1]) .* sin.(π .* p[:, 2]), :, 1)
        mesh = mkmesh_square(9, 9, 2, 0, 1)
        prob = HDGProblem(PoissonEquation(), mesh; bc=Dirichlet(0.0), source)

        sol_d = solve(prob, Direct())
        sol_g = solve(prob, GMRES(tol=1e-10, restart=200))
        sol_u = solve(prob, GMRES(tol=1e-10, restart=200, batched=false))
        @test sol_g.iterations > 0
        @test norm(sol_g.u .- sol_d.u) / norm(sol_d.u) < 1e-7
        @test norm(sol_u.u .- sol_d.u) / norm(sol_d.u) < 1e-7
        @test l2error(sol_d, exact) < 1e-2
        @test size(sol_d.q, 2) == 2
    end

    @testset "named boundary tags" begin
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        @test boundary_names(mesh) == [:bottom, :right, :top, :left]

        eq = ConvectionDiffusionEquation([1.0, 0.5], 0.01)
        u0 = [(x, y) -> exp(-4 * ((x - 0.5)^2 + y^2))]
        bc_vec = [Dirichlet(), Neumann(), Dirichlet(), Neumann()]
        bc_named = (top=Dirichlet(), left=Neumann(), bottom=Dirichlet(), right=Neumann())

        sol_vec = solve(DGProblem(eq, mesh; bc=bc_vec, u0), RK4(); dt=1e-4, nstep=5)
        sol_named = solve(DGProblem(eq, mesh; bc=bc_named, u0), RK4(); dt=1e-4, nstep=5)
        @test sol_named.u == sol_vec.u

        # wrong / missing names are rejected with a helpful error
        bad = DGProblem(eq, mesh; bc=(north=Dirichlet(),), u0)
        @test_throws ArgumentError solve(bad, RK4(); dt=1e-4, nstep=1)
        incomplete = DGProblem(eq, mesh; bc=(top=Dirichlet(), left=Neumann()), u0)
        @test_throws ArgumentError solve(incomplete, RK4(); dt=1e-4, nstep=1)
    end

    @testset "compute_dt (CFL helper)" begin
        mesh = mkmesh_square(9, 9, 3, 0, 1)
        u0 = [(x, y) -> exp(-4 * ((x - 0.5)^2 + y^2))]
        conv = DGProblem(ConvectionEquation([1.0, 0.5]), mesh;
                         bc=fill(FarField([0.0]), 4), u0)
        dt = compute_dt(conv)
        @test dt > 0
        @test compute_dt(conv; cfl=0.15) ≈ dt / 2

        # diffusion tightens the step; finer mesh and higher order tighten it too
        cd = DGProblem(ConvectionDiffusionEquation([1.0, 0.5], 0.1), mesh;
                       bc=fill(Dirichlet(), 4), u0)
        @test compute_dt(cd) < dt
        fine = DGProblem(ConvectionEquation([1.0, 0.5]), mkmesh_square(17, 17, 3, 0, 1);
                         bc=fill(FarField([0.0]), 4), u0)
        @test compute_dt(fine) < dt

        # Euler wave speed comes from the initial state
        γ = 1.4
        uinf = [1.0, 0.3, 0.05, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2)]
        euler = DGProblem(EulerEquations(γ=γ), mesh;
                          bc=fill(FarField(uinf), 4), u0=[(x, y) -> uinf[c] for c in 1:4])
        dte = compute_dt(euler)
        @test 0 < dte < dt   # sound speed ≫ convection speed here

        # the computed step is actually stable for the internal RK4 stepper
        sol = solve(conv, RK4(); dt=compute_dt(conv), tfinal=50 * compute_dt(conv))
        @test all(isfinite, sol.u)
    end

    @testset "semidiscretize -> ODEProblem (A1.5)" begin
        mesh = mkmesh_square(9, 9, 3, 0, 1)
        eq = ConvectionEquation([1.0, 0.5])
        u0 = [(x, y) -> exp(-16 * ((x - 0.5)^2 + (y - 0.5)^2))]
        prob = DGProblem(eq, mesh; bc=fill(FarField([0.0]), 4), u0)

        tfinal = 0.05
        ode = semidiscretize(prob, (0.0, tfinal))
        @test ode isa ODEProblem

        sol_ode = solve(ode, Tsit5(); abstol=1e-10, reltol=1e-10)
        sol_rk4 = solve(prob, RK4(); dt=1e-4, tfinal=tfinal)
        @test sol_ode.t[end] ≈ tfinal
        @test norm(sol_ode.u[end] .- sol_rk4.u) / norm(sol_rk4.u) < 1e-6
    end

    @testset "CGProblem Poisson" begin
        exact(x, y) = sin(π * x) * sin(π * y)
        source(x, y) = 2π^2 * exact(x, y)
        mesh = mkmesh_square(9, 9, 3, 0, 1)
        prob = CGProblem(PoissonEquation(), mesh; source)
        sol = solve(prob)
        @test l2error(sol, exact) < 1e-4
        @test isfinite(sol.energy)
        @test sol.iterations == 0   # direct solve

        # matrix-free iterative paths reproduce the direct solve
        sol_cg = solve(prob, ConjugateGradient(tol=1e-12))
        @test sol_cg.iterations > 0
        @test norm(sol_cg.u .- sol.u) / norm(sol.u) < 1e-8
        @test sol_cg.energy ≈ sol.energy rtol = 1e-8

        cd = CGProblem(ConvectionDiffusionEquation([1.0, 0.5], 1.0), mesh; source)
        @test_throws ArgumentError solve(cd, ConjugateGradient())
        sol_g = solve(cd, GMRES(tol=1e-12))
        sol_d = solve(cd, Direct())
        @test norm(sol_g.u .- sol_d.u) / norm(sol_d.u) < 1e-8
    end
end
