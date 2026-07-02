# High-level problem/solve API: equations-as-structs, boundary-condition
# objects, and the CommonSolve `solve` entry point must reproduce the
# low-level drivers they wrap.

using TwoDG
using TwoDG.Interface: lower_bcs, lower_dbc
using Test
using LinearAlgebra
using StaticArrays

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

    @testset "CGProblem Poisson" begin
        exact(x, y) = sin(π * x) * sin(π * y)
        source(x, y) = 2π^2 * exact(x, y)
        mesh = mkmesh_square(9, 9, 3, 0, 1)
        sol = solve(CGProblem(PoissonEquation(), mesh; source))
        @test l2error(sol, exact) < 1e-4
        @test isfinite(sol.energy)
    end
end
