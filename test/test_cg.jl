# Continuous Galerkin: Poisson with a manufactured solution, p-convergence,
# and the Phase 4 batched/matrix-free path (parity with elemmat_cg, direct
# vs iterative agreement, Float32).

using TwoDG
using TwoDG.ContinuousGalerkin: cg_element_system, cg_assemble, cg_dirichlet_mask
using Test
using LinearAlgebra

@testset "Continuous Galerkin Poisson" begin
    exact(x, y) = sin(π * x) * sin(π * y)
    source(x, y) = 2π^2 * exact(x, y)
    param = (; κ=1.0, c=[0.0, 0.0], s=0.0)

    errs = map((1, 2, 3)) do porder
        mesh = mkmesh_square(9, 9, porder, 0, 1)
        master = Master(mesh, 4porder)
        uh, energy = cg_solve(mesh, master, source, param)
        l2error(mesh, uh, exact)
    end

    @test errs[1] < 3e-2
    @test all(errs[i + 1] < errs[i] / 4 for i in 1:2)  # rapid p-convergence
end

@testset "CG batched path (Phase 4)" begin
    source(x, y) = exp(-2 * ((x - 0.3)^2 + (y - 0.6)^2))
    param = (; κ=0.7, c=[1.5, -0.4], s=0.3)

    @testset "element parity vs elemmat_cg ($name)" for (name, mesh) in (
            ("square p2", mkmesh_square(6, 5, 2, 0, 1)),
            ("distorted square p3", mkmesh_distort!(mkmesh_square(5, 5, 3, 0, 0))))
        master = Master(mesh, 4 * mesh.porder)
        ae, fe, _, _ = cg_element_system(mesh, master, source, param; eliminate=false)
        for e in (1, size(mesh.tcg, 1) ÷ 2, size(mesh.tcg, 1))
            A, F = elemmat_cg(mesh.pcg[mesh.tcg[e, :], :], master, source, param)
            @test ae[:, :, e] ≈ A rtol = 1e-12
            @test fe[:, e] ≈ F rtol = 1e-12
        end
    end

    @testset "symmetric elimination ≡ legacy row-zeroing" begin
        # with homogeneous Dirichlet data, eliminating columns as well as rows
        # must not change the solution — only restore symmetry
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        master = Master(mesh, 8)
        p = (; κ=1.0, c=[0.0, 0.0], s=0.0)
        ae, fe, dirichlet, _ = cg_element_system(mesh, master, source, p)
        K, F = cg_assemble(ae, fe, mesh.tcg, dirichlet)
        # symmetric after elimination (up to BLAS rounding in the element GEMMs)
        @test norm(K - K') <= 1e-13 * norm(K)

        # legacy treatment: zero rows only, unit diagonal per element
        ae2, fe2, _, _ = cg_element_system(mesh, master, source, p; eliminate=false)
        for e in axes(mesh.tcg, 1), i in axes(ae2, 1)
            if dirichlet[mesh.tcg[e, i]]
                ae2[i, :, e] .= 0.0
                ae2[i, i, e] = 1.0
                fe2[i, e] = 0.0
            end
        end
        K2 = zeros(size(K))
        F2 = zeros(length(F))
        for e in axes(mesh.tcg, 1)
            gl = mesh.tcg[e, :]
            K2[gl, gl] .+= ae2[:, :, e]
            F2[gl] .+= fe2[:, e]
        end
        @test K \ F ≈ K2 \ F2 rtol = 1e-12
    end

    @testset "cg_parsolve vs cg_solve ($name)" for (name, p) in (
            ("Poisson (CG)", (; κ=1.0, c=[0.0, 0.0], s=0.0)),
            ("reaction (CG)", (; κ=0.7, c=[0.0, 0.0], s=0.5)),
            ("convection (GMRES)", (; κ=0.7, c=[1.5, -0.4], s=0.3)))
        mesh = mkmesh_square(9, 9, 3, 0, 1)
        master = Master(mesh, 12)
        uh_d, energy_d = cg_solve(mesh, master, source, p)
        uh_i, energy_i, niter = cg_parsolve(mesh, master, source, p; tol=1e-12)
        @test niter > 0
        @test norm(uh_i - uh_d) / norm(uh_d) < 1e-8
        @test energy_i ≈ energy_d rtol = 1e-8

        # unpreconditioned still converges (slower)
        _, _, niter_np = cg_parsolve(mesh, master, source, p; tol=1e-12,
                                     preconditioner=false)
        @test niter_np > 0
    end

    @testset "Float32" begin
        mesh = mkmesh_square(7, 7, 2, 0, 1)
        master = Master(mesh, 8)
        p = (; κ=1.0, c=[0.0, 0.0], s=0.0)
        uh64, _ = cg_solve(mesh, master, source, p)
        uh32, energy32, _ = cg_parsolve(mesh, master, source, p;
                                        T=Float32, tol=1e-6)
        @test eltype(uh32) == Float32
        @test isfinite(energy32)
        @test norm(Float64.(uh32) - uh64) / norm(uh64) < 1e-3
    end

    @testset "boundary mask" begin
        mesh = mkmesh_square(5, 5, 2, 0, 1)
        master = Master(mesh, 8)
        dirichlet = cg_dirichlet_mask(mesh, master)
        onbnd = [any(≈(0; atol=1e-12), mesh.pcg[i, :]) ||
                 any(≈(1; atol=1e-12), mesh.pcg[i, :]) for i in axes(mesh.pcg, 1)]
        @test dirichlet == onbnd
    end
end
