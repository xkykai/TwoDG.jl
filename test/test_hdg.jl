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
        push!(errs_u, sqrt(l2_error(mesh, u[:, 1, :], exact)))
        push!(errs_ustar, sqrt(l2_error(mesh1, ustar[:, 1, :], exact)))
    end

    # O(h^{p+1}) for u; postprocessing gains accuracy on every mesh
    @test log2(errs_u[1] / errs_u[2]) > porder + 0.5
    @test all(errs_ustar .< errs_u ./ 2)

    # matrix-free block-Jacobi GMRES: same solution quality as the direct
    # solve (the two paths pin boundary-touching trace endpoints differently,
    # so they agree to discretization — not solver — accuracy), and the
    # preconditioner must change the iteration count, not the answer
    mesh = mkmesh_square(5, 5, porder, 0, 1)
    master = Master(mesh, ngauss)
    u, _, _ = hdg_solve(master, mesh, source, dbc, param)
    up, _, _, _ = hdg_parsolve(master, mesh, source, dbc, param;
                               restart=200, tol=1e-10)
    upn, _, _, _ = hdg_parsolve(master, mesh, source, dbc, param;
                                restart=200, tol=1e-10, preconditioner=false)
    @test sqrt(l2_error(mesh, up, exact)) < 1.5 * sqrt(l2_error(mesh, u[:, 1, :], exact))
    @test norm(vec(up) .- vec(upn)) / norm(upn) < 1e-8
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

    # full driver matches the legacy handwritten-GMRES path
    u_ka, q_ka, _, niter = hdg_parsolve_ka(master, mesh, source, dbc, param; tol=1e-10, restart=200)
    u_leg, q_leg, _, _ = hdg_parsolve(master, mesh, source, dbc, param; tol=1e-10, restart=200)
    @test niter > 0
    @test norm(u_ka .- u_leg) / norm(u_leg) < 1e-8
    @test norm(q_ka .- q_leg) / norm(q_leg) < 1e-8

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

    # l2_error returns the squared L2 error
    err_u = sqrt(l2_error(mesh, result.u[:, 1, :], u1e) +
                 l2_error(mesh, result.u[:, 2, :], u2e))
    err_p = sqrt(l2_error(mesh, result.p, pe))
    @test err_u < 1e-2
    @test err_p < 1e-2
    # the recovered velocity gradient should be (discretely) divergence free
    @test maximum(abs.(result.gradu[:, 1, :] .+ result.gradu[:, 4, :])) < 1e-8
    # global zero-mean pressure gauge
    @test abs(sum(result.ρ)) < 1e-8

    # divergence-free postprocessing: u* is more accurate than u and its
    # pointwise divergence vanishes
    ustar = hdg_ns_postprocess(master, mesh, master1, mesh1, result)
    err_us = sqrt(l2_error(mesh1, ustar[:, 1, :], u1e) +
                  l2_error(mesh1, ustar[:, 2, :], u2e))
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
