# HDG: Poisson convergence and superconvergent postprocessing, iterative solver
# consistency, and the incompressible Navier-Stokes solver (Kovasznay flow).

using TwoDG
using TwoDG.HybridizableDiscontinuousGalerkin: hdg_elemmats, hdg_matvec,
    apply_blockjacobi, HDGMatVecOp, HDGBlockJacobiOp
using Test
using LinearAlgebra

@testset "HDG Poisson (p = 2)" begin
    exact(x, y) = sin(π * x) * sin(π * y)
    source(p) = reshape(2π^2 .* sin.(π .* p[:, 1]) .* sin.(π .* p[:, 2]), :, 1)
    dbc(p) = zeros(size(p, 1), 1)
    param = Dict(:kappa => 1.0, :c => [0.0, 0.0], :taud => 1.0)

    porder = 2
    ngauss = 4 * (porder + 1)

    errs_u, errs_ustar = Float64[], Float64[]
    for n in (5, 9)
        mesh = mkmesh_square(n, n, porder, 0, 1)
        master = Master(mesh, ngauss)
        mesh1 = mkmesh_square(n, n, porder + 1, 0, 1)
        master1 = Master(mesh1, ngauss)

        u, q, uhat = hdg_solve(master, mesh, source, dbc, param)
        ustar = hdg_postprocess(master, mesh, master1, mesh1, u, q ./ param[:kappa])
        push!(errs_u, l2error(mesh, u[:, 1, :], exact))
        push!(errs_ustar, l2error(mesh1, ustar[:, 1, :], exact))
    end

    # O(h^{p+1}) for u; postprocessing gains accuracy on every mesh
    @test log2(errs_u[1] / errs_u[2]) > porder + 0.5
    @test all(errs_ustar .< errs_u ./ 2)

    # matrix-free block-Jacobi GMRES: direct LU and GMRES now assemble the
    # identical trace system, so they agree to solver accuracy, and the
    # preconditioner must change the iteration count, not the answer
    mesh = mkmesh_square(5, 5, porder, 0, 1)
    master = Master(mesh, ngauss)
    u, _, _ = hdg_solve(master, mesh, source, dbc, param)

    # batched sparse-direct path (the one Interface's Direct() uses) agrees
    # with the per-element reference to solver precision
    ud, _, _ = hdg_direct_batched(master, mesh, source, dbc, param)
    @test norm(vec(ud) .- vec(u)) / norm(vec(u)) < 1e-8

    up, _, _, _ = hdg_parsolve(master, mesh, source, dbc, param;
                               restart=200, tol=1e-10)
    upn, _, _, _ = hdg_parsolve(master, mesh, source, dbc, param;
                                restart=200, tol=1e-10, preconditioner=false)
    @test norm(vec(up) .- vec(u)) / norm(vec(u)) < 1e-7
    @test norm(vec(up) .- vec(upn)) / norm(vec(upn)) < 1e-8
end

@testset "HDG KA/Krylov trace solver (Phase 3)" begin
    # convection-diffusion so the trace system is nonsymmetric
    source(p) = reshape(2π^2 .* sin.(π .* p[:, 1]) .* sin.(π .* p[:, 2]), :, 1)
    dbc(p) = zeros(size(p, 1), 1)
    param = Dict(:kappa => 1.0, :c => [1.0, 0.5], :taud => 1.0)

    porder = 3
    mesh = mkmesh_square(9, 9, porder, 0, 1)
    master = Master(mesh, 4 * (porder + 1))

    ae, fe = hdg_elemmats(master, mesh, source, dbc, param)
    sys = HDGSystem(ae, fe, mesh)

    # KA kernels reproduce the legacy threaded matvec / block-Jacobi apply
    x = randn(length(sys.b))
    y = similar(x)
    mul!(y, HDGMatVecOp(sys), x)
    @test y ≈ hdg_matvec(sys.A, x, mesh.f2f) rtol = 1e-13
    mul!(y, HDGBlockJacobiOp(sys), x)
    @test y ≈ apply_blockjacobi(sys.B, x) rtol = 1e-13

    # Krylov GMRES: preconditioned and unpreconditioned solve the same system
    xp, stats = hdg_gmres_ka(sys; tol=1e-10, restart=200)
    @test stats.solved
    xnp, _ = hdg_gmres_ka(sys; tol=1e-10, restart=200, preconditioner=false)
    @test norm(xp .- xnp) / norm(xnp) < 1e-6

    # full iterative driver matches the direct (sparse LU) solve of the same
    # trace system — a real assembly + solver consistency check
    u_ka, q_ka, _, niter = hdg_parsolve(master, mesh, source, dbc, param; tol=1e-10, restart=200)
    u_dir, q_dir, _ = hdg_solve(master, mesh, source, dbc, param)
    @test niter > 0
    @test norm(u_ka .- u_dir) / norm(u_dir) < 1e-7
    @test norm(q_ka .- q_dir) / norm(q_dir) < 1e-7

    # Float32 system solves and stays close to the Float64 solution
    sys32 = HDGSystem(ae, fe, mesh; T=Float32)
    x32, stats32 = hdg_gmres_ka(sys32; tol=1e-5)
    @test eltype(x32) == Float32 && stats32.solved
    @test norm(Float64.(x32) .- xp) / norm(xp) < 1e-3
end

@testset "HDG batched assembly + recovery (Phase 3b)" begin
    source(p) = reshape(sin.(π .* p[:, 1]) .* p[:, 2], :, 1)
    dbc(p) = 0.1 .* p[:, 1:1] .- 0.2 .* p[:, 2:2]
    # taud ≠ kappa exercises the localprob (tau = κ + |c·n|) vs elemmat_hdg
    # (tau = taud + |c·n|) distinction the batch must replicate
    param = Dict(:kappa => 0.8, :c => [1.0, 0.5], :taud => 1.2)

    # batched ae/fe match the legacy per-element assembly (straight + curved)
    for mesh in (mkmesh_square(7, 7, 3, 0, 1), mkmesh_trefftz(8, 16, 3))
        master = Master(mesh, 4 * (mesh.porder + 1))
        loc = hdg_local_solves(HDGBatch(master, mesh, source, param))
        ae_ref = similar(loc.ae)
        fe_ref = similar(loc.fe)
        for e in axes(mesh.dgnodes, 3)
            a, f = elemmat_hdg(mesh.dgnodes[:, :, e], master, source, param)
            ae_ref[:, :, e] .= a
            fe_ref[:, e] .= f
        end
        @test norm(loc.ae .- ae_ref) / norm(ae_ref) < 1e-9
        @test norm(loc.fe .- fe_ref) / norm(fe_ref) < 1e-9
    end

    # end-to-end: batched driver (assembly + GMRES + batched recovery) matches
    # the legacy hdg_parsolve solution
    mesh = mkmesh_square(9, 9, 3, 0, 1)
    master = Master(mesh, 16)
    u_b, q_b, _, _ = hdg_parsolve_batched(master, mesh, source, dbc, param;
                                          tol=1e-10, restart=200)
    u_leg, q_leg, _, _ = hdg_parsolve(master, mesh, source, dbc, param;
                                      tol=1e-10, restart=200)
    @test norm(u_b .- u_leg) / norm(u_leg) < 1e-8
    @test norm(q_b .- q_leg) / norm(q_leg) < 1e-8
end

@testset "HDG incompressible Navier-Stokes (Kovasznay flow)" begin
    # Kovasznay flow at Re = 20 on (0,2) x (-0.5,1.5): an exact solution of
    # the homogeneous steady incompressible Navier-Stokes equations.
    Re = 20.0
    ν = 1 / Re
    λk = Re / 2 - sqrt(Re^2 / 4 + 4π^2)
    u1e(x, y) = 1 - exp(λk * x) * cos(2π * y)
    u2e(x, y) = λk / (2π) * exp(λk * x) * sin(2π * y)
    pmean = -(exp(4λk) - 1) / (8λk)
    pe(x, y) = -exp(2λk * x) / 2 - pmean
    dbc(p) = [u1e(p[1], p[2]), u2e(p[1], p[2])]

    n, porder = 8, 2
    mesh = mkmesh_square(n + 1, n + 1, porder, 0, 1)
    for arr in (mesh.p, mesh.pcg)
        arr[:, 1] .= 2 .* arr[:, 1]
        arr[:, 2] .= 2 .* arr[:, 2] .- 0.5
    end
    mesh.dgnodes[:, 1, :] .= 2 .* mesh.dgnodes[:, 1, :]
    mesh.dgnodes[:, 2, :] .= 2 .* mesh.dgnodes[:, 2, :] .- 0.5
    master = Master(mesh, 3 * (porder + 1))

    mesh1 = mkmesh_square(n + 1, n + 1, porder + 1, 0, 1)
    for arr in (mesh1.p, mesh1.pcg)
        arr[:, 1] .= 2 .* arr[:, 1]
        arr[:, 2] .= 2 .* arr[:, 2] .- 0.5
    end
    mesh1.dgnodes[:, 1, :] .= 2 .* mesh1.dgnodes[:, 1, :]
    mesh1.dgnodes[:, 2, :] .= 2 .* mesh1.dgnodes[:, 2, :] .- 0.5
    master1 = Master(mesh1, 3 * (porder + 1))

    result = hdg_ns_solve(master, mesh, ν, dbc; τ=1.0, maxiter=12, tol=1e-10, verbose=false)

    err_u = hypot(l2error(mesh, result.u[:, 1, :], u1e),
                  l2error(mesh, result.u[:, 2, :], u2e))
    err_p = l2error(mesh, result.p, pe)
    @test err_u < 1e-2
    @test err_p < 1e-2
    # the recovered velocity gradient should be (discretely) divergence free
    @test maximum(abs.(result.gradu[:, 1, :] .+ result.gradu[:, 4, :])) < 1e-8
    # global zero-mean pressure gauge
    @test abs(sum(result.ρ)) < 1e-8

    # divergence-free postprocessing: u* is more accurate than u and its
    # pointwise divergence vanishes
    ustar = hdg_ns_postprocess(master, mesh, master1, mesh1, result)
    err_us = hypot(l2error(mesh1, ustar[:, 1, :], u1e),
                   l2error(mesh1, ustar[:, 2, :], u2e))
    @test err_us < err_u / 2
    maxdiv = 0.0
    sh = master1.shap
    for it in axes(mesh1.dgnodes, 3)
        dg = mesh1.dgnodes[:, :, it]
        xxi = sh[:, 2, :]' * dg[:, 1]; xet = sh[:, 3, :]' * dg[:, 1]
        yxi = sh[:, 2, :]' * dg[:, 2]; yet = sh[:, 3, :]' * dg[:, 2]
        jac = xxi .* yet .- xet .* yxi
        d1 = (sh[:, 2, :]' * ustar[:, 1, it] .* yet .- sh[:, 3, :]' * ustar[:, 1, it] .* yxi) ./ jac
        d2 = (-sh[:, 2, :]' * ustar[:, 2, it] .* xet .+ sh[:, 3, :]' * ustar[:, 2, it] .* xxi) ./ jac
        maxdiv = max(maxdiv, maximum(abs.(d1 .+ d2)))
    end
    @test maxdiv < 1e-7
end

@testset "HDG NS/CD batched assembly + drivers (Phase 5)" begin
    relerr(a, b) = norm(a .- b) / max(norm(b), eps())

    # Kovasznay boundary data (nontrivial Dirichlet trace)
    Re = 20.0
    ν = 1 / Re
    λk = Re / 2 - sqrt(Re^2 / 4 + 4π^2)
    dbc(p) = [1 - exp(λk * p[1]) * cos(2π * p[2]),
              λk / (2π) * exp(λk * p[1]) * sin(2π * p[2])]

    # on the airfoil domain the Kovasznay data is meaningless and the first
    # Newton iterate blows up (|u| ~ 1e8, λ²-sized matrix entries ~1e16, so a
    # roundoff-level comparison is impossible) — use freestream data there
    @testset "NS step parity ($name)" for (name, mesh, ν, dbc) in
            (("square", mkmesh_square(7, 7, 2, 0, 1), ν, dbc),
             ("curved trefftz", mkmesh_trefftz(8, 16, 2), 0.05, p -> [1.0, 0.0]))
        porder = mesh.porder
        master = Master(mesh, 3 * (porder + 1))
        npl, nt = size(mesh.dgnodes, 1), size(mesh.t, 1)

        # zero-state step (fresh cache, Dirichlet projection path); p/ρ/L are
        # measured against the velocity scale — for freestream data they are
        # analytically zero, so a self-relative error is noise over noise
        s1 = hdg_ns_step(master, mesh, ν, dbc; τ=1.0)
        b1 = hdg_ns_step_batched(master, mesh, ν, dbc; τ=1.0)
        scale = norm(s1.u)
        @test relerr(b1.u, s1.u) < 1e-8
        @test norm(b1.p .- s1.p) / scale < 1e-8
        @test relerr(b1.Λ, s1.Λ) < 1e-8
        @test norm(b1.gradu .- s1.gradu) / scale < 1e-7
        @test norm(b1.ρ .- s1.ρ) / scale < 1e-8

        # nonzero linearization state + array source + backward Euler,
        # exercising the cached pattern and numeric refactorization (lu!)
        src = zeros(npl, 2, nt)
        src[:, 2, :] .= sin.(mesh.dgnodes[:, 1, :])
        dtinv = 2.0
        s2 = hdg_ns_step(master, mesh, ν, dbc; τ=1.0, source=src,
                         u=s1.u, Λ=s1.Λ, uold=s1.u, dtinv)
        b2 = hdg_ns_step_batched(master, mesh, ν, dbc; τ=1.0, source=src,
                                 u=s1.u, Λ=s1.Λ, uold=s1.u, dtinv,
                                 cache=b1.cache)
        @test relerr(b2.u, s2.u) < 1e-8
        @test relerr(b2.p, s2.p) < 1e-8
        @test relerr(b2.Λ, s2.Λ) < 1e-8
        @test relerr(b2.gradu, s2.gradu) < 1e-7

        # function source (quadrature-point evaluation path)
        ffun(p) = [p[2], -p[1]]
        s3 = hdg_ns_step(master, mesh, ν, dbc; τ=1.0, source=ffun,
                         u=s1.u, Λ=s1.Λ)
        b3 = hdg_ns_step_batched(master, mesh, ν, dbc; τ=1.0, source=ffun,
                                 u=s1.u, Λ=s1.Λ, cache=b1.cache)
        @test relerr(b3.u, s3.u) < 1e-8
        @test relerr(b3.Λ, s3.Λ) < 1e-8
    end

    @testset "CD step parity" begin
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        master = Master(mesh, 3 * (mesh.porder + 1))
        npl, nt = size(mesh.dgnodes, 1), size(mesh.t, 1)
        κ = 0.05
        # Boussinesq-style mixed BCs: hot/cold Dirichlet walls + insulated
        tbc(p, tag) = tag == 4 ? (:d, 0.5) : tag == 2 ? (:d, -0.5) : (:n, 0.0)

        ns = hdg_ns_step(master, mesh, ν, dbc; τ=1.0)   # convecting field
        θold = [0.5 - mesh.dgnodes[k, 1, it] for k in 1:npl, it in 1:nt]

        s1 = hdg_cd_step(master, mesh, κ, tbc; τ=1.0, u=ns.u, Λ=ns.Λ,
                         θold, dtinv=4.0)
        b1 = hdg_cd_step_batched(master, mesh, κ, tbc; τ=1.0, u=ns.u, Λ=ns.Λ,
                                 θold, dtinv=4.0)
        @test relerr(b1.θ, s1.θ) < 1e-8
        @test relerr(b1.q, s1.q) < 1e-7
        @test relerr(b1.Θ, s1.Θ) < 1e-8

        # cache reuse at a new state + array source
        src = cos.(mesh.dgnodes[:, 2, :])
        s2 = hdg_cd_step(master, mesh, κ, tbc; τ=1.0, u=ns.u, Λ=ns.Λ,
                         θold=s1.θ, dtinv=4.0, source=src)
        b2 = hdg_cd_step_batched(master, mesh, κ, tbc; τ=1.0, u=ns.u, Λ=ns.Λ,
                                 θold=b1.θ, dtinv=4.0, source=src,
                                 cache=b1.cache)
        @test relerr(b2.θ, s2.θ) < 1e-8
        @test relerr(b2.Θ, s2.Θ) < 1e-8
    end

    @testset "Boussinesq mini-cavity trajectory" begin
        # 5 operator-splitting steps of the heated cavity on both paths
        Ra, Pr = 1e3, 0.71
        νb = sqrt(Pr / Ra)
        κb = 1 / sqrt(Ra * Pr)
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        master = Master(mesh, 3 * (mesh.porder + 1))
        npl, nt = size(mesh.dgnodes, 1), size(mesh.t, 1)
        nps, nf = mesh.porder + 1, size(mesh.f, 1)
        dbc0(p) = [0.0, 0.0]
        tbc(p, tag) = tag == 4 ? (:d, 0.5) : tag == 2 ? (:d, -0.5) : (:n, 0.0)
        dtinv = 1 / 0.5

        θl = [0.5 - mesh.dgnodes[k, 1, it] for k in 1:npl, it in 1:nt]
        ul = zeros(npl, 2, nt)
        Λl = zeros(2 * nps * nf)
        θb, ub, Λb = copy(θl), copy(ul), copy(Λl)
        nscache = nothing
        cdcache = nothing
        for step in 1:5
            θres = hdg_cd_step(master, mesh, κb, tbc; τ=1.0, u=ul, Λ=Λl,
                               θold=θl, dtinv)
            θl = θres.θ
            srcl = zeros(npl, 2, nt)
            srcl[:, 2, :] .= θl
            uoldl = copy(ul)
            for inner in 1:2
                resl = hdg_ns_step(master, mesh, νb, dbc0; τ=1.0, source=srcl,
                                   u=ul, Λ=Λl, uold=uoldl, dtinv)
                ul, Λl = resl.u, resl.Λ
            end

            θresb = hdg_cd_step_batched(master, mesh, κb, tbc; τ=1.0, u=ub,
                                        Λ=Λb, θold=θb, dtinv, cache=cdcache)
            θb = θresb.θ
            cdcache = θresb.cache
            srcb = zeros(npl, 2, nt)
            srcb[:, 2, :] .= θb
            uoldb = copy(ub)
            for inner in 1:2
                resb = hdg_ns_step_batched(master, mesh, νb, dbc0; τ=1.0,
                                           source=srcb, u=ub, Λ=Λb,
                                           uold=uoldb, dtinv, cache=nscache)
                ub, Λb = resb.u, resb.Λ
                nscache = resb.cache
            end
        end
        @test relerr(θb, θl) < 1e-7
        @test relerr(ub, ul) < 1e-7
    end

    @testset "Float32 sanity" begin
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        master = Master(mesh, 3 * (mesh.porder + 1))
        s = hdg_ns_step(master, mesh, ν, dbc; τ=1.0)
        b = hdg_ns_step_batched(master, mesh, ν, dbc; τ=1.0, T=Float32)
        @test relerr(b.u, s.u) < 1e-2
    end
end
