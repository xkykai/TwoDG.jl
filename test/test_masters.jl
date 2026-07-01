# Master element: quadrature rules, Koornwinder bases, nodal shape functions.

using TwoDG
using Test
using LinearAlgebra

@testset "Master element" begin
    @testset "1D Gauss quadrature integrates degree $d exactly" for d in (1, 4, 7, 12)
        x, w = gaussquad1d(d)
        @test all(abs(sum(w .* x .^ j) - 1 / (j + 1)) < 1e-13 for j in 0:d)
    end

    @testset "2D Gauss quadrature integrates degree $d exactly" for d in (2, 5, 10)
        x, w = gaussquad2d(d)
        # ∫_T ξ^a η^b dA = a! b! / (a+b+2)! on the unit triangle
        exact(a, b) = Float64(factorial(big(a)) * factorial(big(b)) //
                              factorial(big(a + b + 2)))
        @test all(abs(sum(w .* x[:, 1] .^ a .* x[:, 2] .^ b) - exact(a, b)) < 1e-12
                  for a in 0:d for b in 0:(d - a))
    end

    @testset "Koornwinder bases are orthonormal (p = $p)" for p in (2, 5)
        x1, w1 = gaussquad1d(2p + 1)
        f1, _ = koornwinder1d(x1, p)
        @test norm(f1' * Diagonal(w1) * f1 - I) < 1e-12

        x2, w2 = gaussquad2d(2p + 1)
        V, _, _ = koornwinder2d(x2, p)
        @test norm(V' * Diagonal(w2) * V - I) < 1e-12
    end

    @testset "shape functions and mass matrix (p = $p)" for p in (1, 3, 5)
        mesh = mkmesh_square(3, 3, p, 0, 1)
        master = Master(mesh)

        # partition of unity and zero derivative sums, at all quadrature points
        @test maximum(abs, sum(master.shap[:, 1, :], dims=1) .- 1) < 1e-12
        @test maximum(abs, sum(master.shap[:, 2:3, :], dims=1)) < 1e-10
        @test maximum(abs, sum(master.sh1d[:, 1, :], dims=1) .- 1) < 1e-12

        # reference mass matrix: symmetric positive definite, ∫_T 1 dA = 1/2
        @test norm(master.mass - master.mass') < 1e-14
        @test isposdef(Symmetric(master.mass))
        @test abs(sum(master.mass) - 0.5) < 1e-12

        # perm[:, k, 1] lists the nodes on local face k (k-th barycentric coord 0)
        @test all(all(master.plocal[master.perm[:, k, 1], k] .< 1e-6) for k in 1:3)
        @test master.perm[:, :, 2] == reverse(master.perm[:, :, 1], dims=1)
    end
end
