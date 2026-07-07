# The run-time callback layer (CALLBACKS_PLAN.md): schedules are pure logic,
# diagnostics are quadrature-exact, and callbacks are observers — a run with
# callbacks attached must produce a bit-identical solution to a run without.

using TwoDG
using TwoDG.Callbacks: initialize!, finish!, load_checkpoint
using Test
using LinearAlgebra
using StaticArrays
using Serialization: deserialize

# a physics-free state for driving schedules and non-firing callbacks
_dummy_state(; t=0.0, step=0, dt=0.1) =
    SolveState(nothing, t, step, dt, 100, NaN, nothing, nothing, nothing)

@testset "Callbacks & diagnostics" begin
    @testset "schedules fire on the right steps" begin
        st = _dummy_state()

        it = IterationInterval(3)
        @test [k for k in 1:10 if (st.step = k; it(st))] == [3, 6, 9]
        @test_throws ArgumentError IterationInterval(0)
        @test all(k -> (st.step = k; EveryStep()(st)), 1:5)

        ti = TimeInterval(0.25)
        st = _dummy_state()
        initialize!(ti, st)
        fired = [k for k in 1:10 if (st.step = k; st.t = k * 0.1; ti(st))]
        @test fired == [3, 5, 8, 10]        # crossings of 0.25, 0.5, 0.75, 1.0

        # a step crossing several multiples fires once, then realigns
        ti2 = TimeInterval(0.1)
        st = _dummy_state()
        initialize!(ti2, st)
        st.t = 0.35
        @test ti2(st)
        st.t = 0.38
        @test !ti2(st)                       # next is 0.4 now, not 0.2
        @test_throws ArgumentError TimeInterval(0.0)

        sp = SpecifiedTimes(0.45, 0.15, 99.0)   # sorted internally; 99 never fires
        st = _dummy_state()
        initialize!(sp, st)
        fired = [k for k in 1:10 if (st.step = k; st.t = k * 0.1; sp(st))]
        @test fired == [2, 5]

        wt = WallTimeInterval(0.0)           # zero interval: fires every call
        st = _dummy_state()
        initialize!(wt, st)
        @test wt(st) && wt(st)
        never = WallTimeInterval(1e6)
        initialize!(never, st)
        @test !never(st)
    end

    @testset "CallbackSet runs all members, in order" begin
        calls = Symbol[]
        cs = CallbackSet(st -> (push!(calls, :a); nothing),
                         st -> (push!(calls, :b); true),
                         st -> (push!(calls, :c); false))
        @test cs(_dummy_state()) === true
        @test calls == [:a, :b, :c]          # :c still ran after :b said stop
        @test CallbackSet(st -> nothing, st -> false)(_dummy_state()) === false
    end

    @testset "schedule checks and non-firing callbacks are allocation-free" begin
        check(s, st) = s(st)
        st = _dummy_state(step=7)            # 7 % 10 != 0: nothing fires
        sched = IterationInterval(10)
        pcb = ProgressCallback(interval=10, io=devnull)
        check(sched, st)
        pcb(st)
        @test @allocated(check(sched, st)) == 0
        @test @allocated(pcb(st)) == 0
    end

    @testset "pointwise functionals and CFL primitives" begin
        eq = EulerEquations(γ=1.4)
        u = SVector(1.0, 0.3, 0.05, 2.0)
        @test energy_kinetic(eq, u) ≈ (0.3^2 + 0.05^2) / 2
        @test energy_total(eq, u) == 2.0
        @test energy_internal(eq, u) ≈ 2.0 - energy_kinetic(eq, u)
        @test wavespeed(eq, u) ≈ norm(velocity(eq, u)) + soundspeed(eq, u)

        @test wavespeed(ConvectionEquation([3.0, 4.0]), SVector(1.0)) ≈ 5.0
        @test_throws ArgumentError wavespeed(ConvectionEquation(x -> SVector(-x[2], x[1])),
                                             SVector(1.0))
        @test wavespeed(WaveEquation(-2.0), SVector(0.0, 0.0, 1.0)) == 2.0
        @test diffusivity(PoissonEquation(2.5)) == 2.5
        @test diffusivity(ConvectionDiffusionEquation([1.0, 0.0], 0.01)) == 0.01
        @test diffusivity(ConvectionEquation([1.0, 0.0])) == 0.0

        # 2x2-block unit square: right triangles with legs 1/2
        mesh = mkmesh_square(3, 3, 1, 0, 1)
        @test min_inscribed_diameter(mesh) ≈ 0.5 * (2 - sqrt(2))
    end

    @testset "integrate / l2norm are quadrature-exact (straight + curved)" begin
        eq = ConvectionEquation([1.0, 0.0])
        for mesh in (mkmesh_square(9, 9, 3, 0, 1), mkmesh_trefftz(8, 16, 3))
            ctx = DGContext(Master(mesh), mesh)
            ones_u = initu(mesh, 1, [1.0])
            # constant field: ∫ 1 dx must equal the summed quadrature measure
            # (exercises the affine and the curved VolumeTables paths alike)
            @test integrate((e, v) -> v[1], eq, ones_u, ctx) ≈ sum(ctx.wjac) rtol = 1e-12
            @test integrate(ones_u, ctx)[1] ≈ sum(ctx.wjac) rtol = 1e-12
        end

        mesh = mkmesh_square(9, 9, 3, 0, 1)
        ctx = DGContext(Master(mesh), mesh)
        # tolerance is the p=3 *interpolation* error of the initial data; the
        # quadrature itself is exact to round-off (constant-field tests above)
        u = initu(mesh, 1, [(x, y) -> sin(π * x) * sin(π * y)])
        @test integrate((e, v) -> v[1]^2, eq, u, ctx) ≈ 0.25 rtol = 1e-4
        @test integrate(u, ctx)[1] ≈ 4 / π^2 rtol = 1e-4
        @test l2norm(u, ctx) ≈ 0.5 rtol = 1e-4
        @test l2norm(u, ctx; component=1) == l2norm(u, ctx)
        # ‖u‖ through the context quadrature == legacy l2error against zero
        @test l2norm(u, ctx) ≈ l2error(mesh, u[:, 1, :], (x, y) -> 0.0) rtol = 1e-10
    end

    @testset "callbacks are observers: bit-identical solution, drift ≈ 0" begin
        γ = 1.4
        uinf = [1.0, 0.3, 0.05, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2)]
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        prob = DGProblem(EulerEquations(γ=γ), mesh; bc=fill(FarField(uinf), 4),
                         u0=[(x, y) -> uinf[c] for c in 1:4])

        acb = AnalysisCallback(interval=5, io=devnull,
                               integrals=(ek=energy_kinetic, s=entropy))
        cbs = CallbackSet(ProgressCallback(interval=5, io=devnull), acb)
        sol = solve(prob, RK4(); dt=1e-3, nstep=10, callback=cbs)
        sol0 = solve(prob, RK4(); dt=1e-3, nstep=10)
        @test sol.u == sol0.u                # the contract: observers change no bits
        @test sol.callbacks === cbs          # history rides on the solution

        @test acb.steps == [0, 5, 10]        # t0 row + schedule + final
        @test acb.time ≈ [0.0, 5e-3, 1e-2]
        @test maximum(acb.data[:conservation_drift]) < 1e-11
        @test length(acb.data[:ek]) == 3
        @test acb.data[:min_ρ][1] ≈ 1.0 && acb.data[:max_ρ][1] ≈ 1.0
    end

    @testset "AnalysisCallback L2 error matches post-hoc l2error" begin
        mesh = mkmesh_square(9, 9, 3, 0, 1)
        eq = ConvectionEquation([1.0, 0.5])
        f(x, y) = exp(-16 * ((x - 0.4)^2 + (y - 0.4)^2))
        prob = DGProblem(eq, mesh; bc=fill(FarField([0.0]), 4), u0=[f])
        exact(x, t) = f(x[1] - t, x[2] - 0.5t)   # translated initial condition

        acb = AnalysisCallback(interval=10, errors=(exact,), io=devnull)
        dt, nstep = 1e-3, 20
        sol = solve(prob, RK4(); dt, nstep, callback=acb)
        err_cb = acb.data[:l2error_u][end]
        err_ref = l2error(mesh, sol.u[:, 1, :],
                          (x, y) -> exact(SVector(x, y), nstep * dt))
        @test err_cb ≈ err_ref rtol = 1e-10
        @test err_cb < 1e-2                  # and the solve itself is accurate
    end

    @testset "SteadyStateCallback stops a steady run early" begin
        γ = 1.4
        uinf = [1.0, 0.3, 0.05, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2)]
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        prob = DGProblem(EulerEquations(γ=γ), mesh; bc=fill(FarField(uinf), 4),
                         u0=[(x, y) -> uinf[c] for c in 1:4])
        # free stream is exactly steady: the rate check trips at the first firing
        sol = solve(prob, RK4(); dt=1e-3, nstep=100,
                    callback=SteadyStateCallback(interval=2, abstol=1e-8))
        @test sol.t ≈ 2e-3
    end

    @testset "SaveSolutionCallback round-trips snapshots" begin
        γ = 1.4
        uinf = [1.0, 0.3, 0.05, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2)]
        mesh = mkmesh_square(5, 5, 2, 0, 1)
        eq = EulerEquations(γ=γ)
        prob = DGProblem(eq, mesh; bc=fill(FarField(uinf), 4),
                         u0=[(x, y) -> uinf[c] for c in 1:4])

        dir = mktempdir()
        scb = SaveSolutionCallback(interval=2, path=dir, fields=(:u, :p => pressure))
        sol = solve(prob, RK4(); dt=1e-3, nstep=4, callback=scb)
        @test length(scb.files) == 3         # steps 0, 2, 4 (final not duplicated)
        snap = deserialize(scb.files[end])
        @test snap.step == 4 && snap.t ≈ 4e-3
        @test snap.u == sol.u                # host copy is bit-exact
        @test snap.p ≈ derived_field(pressure, eq, sol.u)
        @test_throws ArgumentError SaveSolutionCallback(fields=(:u, pressure))
    end

    @testset "CheckpointCallback + restart reproduces an uninterrupted run" begin
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        eq = ConvectionEquation([1.0, 0.5])
        u0 = [(x, y) -> exp(-16 * ((x - 0.5)^2 + (y - 0.5)^2))]
        prob = DGProblem(eq, mesh; bc=fill(FarField([0.0]), 4), u0)
        dt = 1e-3

        ref = solve(prob, RK4(); dt, nstep=20)
        path = joinpath(mktempdir(), "chk.jls")
        solve(prob, RK4(); dt, nstep=10,
              callback=CheckpointCallback(path=path, schedule=IterationInterval(10)))
        chk = load_checkpoint(path)
        @test chk.step == 10 && chk.t ≈ 10dt

        sol = solve(prob, RK4(); dt, nstep=10, restart=path)
        @test sol.t ≈ 20dt
        @test norm(sol.u .- ref.u) / norm(ref.u) < 1e-12
    end

    @testset "StepsizeCallback drives the loop at the CFL step" begin
        mesh = mkmesh_square(9, 9, 3, 0, 1)
        prob = DGProblem(ConvectionEquation([1.0, 0.5]), mesh;
                         bc=fill(FarField([0.0]), 4),
                         u0=[(x, y) -> exp(-16 * ((x - 0.5)^2 + (y - 0.5)^2))])
        dts = Float64[]
        tfinal = 0.02
        # the dt keyword is a placeholder: StepsizeCallback owns state.dt
        sol = solve(prob, RK4(); dt=1e9, tfinal,
                    callback=CallbackSet(StepsizeCallback(cfl=0.3),
                                         st -> (push!(dts, st.dt); nothing)))
        @test all(isfinite, sol.u)
        @test sol.t ≈ tfinal                 # last step clamped onto tfinal
        @test dts[1] ≈ compute_dt(prob; cfl=0.3) rtol = 1e-12
    end

    @testset "hdg_ns_solve Newton-iteration callback" begin
        mesh = mkmesh_square(4, 4, 2, 0, 1)
        master = Master(mesh, 9)
        dbc(p) = [0.0, 0.0]                  # rest: Stokes step then converged
        iters, res = Int[], Float64[]
        result = hdg_ns_solve(master, mesh, 1.0, dbc; verbose=false, maxiter=5,
                              callback=st -> (push!(iters, st.iter);
                                              push!(res, st.residual); false))
        @test iters == [1, 2]
        @test res[2] < 1e-12
        @test maximum(abs.(result.u)) < 1e-10

        # returning true stops the Newton loop
        stopped = Int[]
        hdg_ns_solve(master, mesh, 1.0, dbc; verbose=false, maxiter=5,
                     callback=st -> (push!(stopped, st.iter); true))
        @test stopped == [1]
    end
end
