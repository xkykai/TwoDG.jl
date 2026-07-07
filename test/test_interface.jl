# High-level problem/solve API: equations-as-structs, boundary-condition
# objects, and the CommonSolve `solve` entry point must reproduce the
# low-level drivers they wrap.

using TwoDG
using TwoDG.Interface: lower_dbc
using Test
using LinearAlgebra
using StaticArrays
using SciMLBase: ODEProblem
using OrdinaryDiffEqTsit5: Tsit5

# user-defined equation exercising the extension contract through the
# high-level API (top level: structs cannot be defined inside a @testset)
struct InterfaceBurgers <: TwoDG.AbstractEquation{2} end
TwoDG.nvariables(::InterfaceBurgers) = 1
TwoDG.varnames(::InterfaceBurgers) = (:u,)
TwoDG.flux(::InterfaceBurgers, u::SVector{1}, x, t) = (u .* u ./ 2, u .* u ./ 2)
TwoDG.max_abs_speed(::InterfaceBurgers, u::SVector{1}, n, x, t) =
    abs(u[1] * (n[1] + n[2]))
TwoDG.wavespeed(::InterfaceBurgers, u::SVector{1}) = sqrt(2 * one(u[1])) * abs(u[1])

@testset "Interface (problems + solve)" begin
    @testset "boundary-condition objects" begin
        # ghost states by dispatch
        eq = EulerEquations(γ=1.4)
        uinf = SVector(1.0, 0.3, 0.05, 2.0)
        uL = SVector(1.1, 0.2, 0.1, 2.1)
        n = SVector(1.0, 0.0)
        x = SVector(0.0, 0.0)
        @test boundary_state(FarField(uinf), eq, uL, n, x, 0.0) == uinf
        wall = boundary_state(SlipWall(), eq, uL, n, x, 0.0)
        @test wall[2] ≈ -uL[2] && wall[1] == uL[1] && wall[4] == uL[4]
        @test boundary_state(Neumann(), eq, uL, n, x, 0.0) == uL
        @test boundary_state(Dirichlet(0.5), eq, SVector(1.0), n, x, 0.0) == SVector(0.5)
        # (x, t)-dependent Dirichlet data
        g = Dirichlet((x, t) -> x[1] + 2x[2])
        @test boundary_state(g, eq, SVector(1.0), n, SVector(1.0, 2.0), 0.0) == SVector(5.0)

        # HDG Dirichlet-data lowering ((x, y) closure -> coordinate-matrix form)
        dbc = lower_dbc(Dirichlet((x, y) -> x + 2y))
        @test dbc([1.0 2.0; 3.0 4.0]) ≈ [5.0, 11.0]
        @test lower_dbc(Dirichlet(0.5))([1.0 2.0]) == [0.5]

        # wrong boundary count is rejected at solve time
        mesh = mkmesh_square(5, 5, 1, 0, 1)
        bad = DGProblem(ConvectionEquation([1.0, 0.0]), mesh;
                        bc=[FarField([0.0])], u0=[(x, y) -> 0.0])
        @test_throws ArgumentError solve(bad, RK4(); dt=1e-3, nstep=1)
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
        uinf_arr = initu(mesh, 4, u0funcs)
        @test norm(sol.u .- uinf_arr) / norm(uinf_arr) < 1e-10

        # identical to driving the KA stepper by hand
        master = Master(mesh)
        phys = DGPhysics(eq; boundary_conditions=ntuple(_ -> FarField(uinf), 4))
        u_ref = rk4_ka!(DGContext(master, mesh), phys, copy(uinf_arr), 0.0, 1e-3, 5)
        @test sol.u ≈ u_ref rtol = 1e-12
    end

    @testset "DGProblem LDG (convection-diffusion) runs and decays" begin
        eq = ConvectionDiffusionEquation(x -> SVector(-x[2], x[1]), 0.01)
        mesh = mkmesh_square(9, 9, 3, 0, 1)
        u0 = [(x, y) -> exp(-4 * ((x - 0.5)^2 + y^2))]
        prob = DGProblem(eq, mesh; bc=[Dirichlet(), Neumann(), Dirichlet(), Neumann()],
                         u0, stabilization=LDGStabilization(10.0, 0.5))
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

    @testset "solve callback hook" begin
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        eq = ConvectionEquation([1.0, 0.5])
        u0 = [(x, y) -> exp(-16 * ((x - 0.5)^2 + (y - 0.5)^2))]
        prob = DGProblem(eq, mesh; bc=fill(FarField([0.0]), 4), u0)

        # called every step with (u, t, step, prob); norms recorded
        ts = Float64[]
        cb = state -> (push!(ts, state.t); false)
        sol = solve(prob, RK4(); dt=1e-3, nstep=5, callback=cb)
        @test ts ≈ collect(1:5) .* 1e-3
        # callback path gives the same answer as the plain path
        sol0 = solve(prob, RK4(); dt=1e-3, nstep=5)
        @test sol.u == sol0.u

        # returning true stops the loop early
        sol_stop = solve(prob, RK4(); dt=1e-3, nstep=100,
                         callback=state -> state.step >= 3)
        @test sol_stop.t ≈ 3e-3
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

        # user equations reach compute_dt through the generic wavespeed
        # fallback: same mesh and cfl as `conv`, so the steps differ exactly
        # by the speed ratio (u0 peaks at 1 on a mesh node, λ = √2·1)
        ub = DGProblem(InterfaceBurgers(), mesh;
                       bc=fill(FarField([0.0]), 4), u0,
                       numerical_flux=LaxFriedrichs())
        @test compute_dt(ub) ≈ dt * norm([1.0, 0.5]) / sqrt(2) rtol = 1e-12

        # an explicit numerical_flux means default_numerical_flux (which this
        # equation deliberately does not define) is never called
        solb = solve(ub, RK4(); dt=compute_dt(ub), nstep=5)
        @test all(isfinite, solb.u)
    end

    @testset "semidiscretize -> ODEProblem" begin
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
