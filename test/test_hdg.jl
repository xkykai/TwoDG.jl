# HDG: Poisson convergence and superconvergent postprocessing, iterative solver
# consistency, and the incompressible Navier-Stokes solver (Kovasznay flow).

using TwoDG
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
