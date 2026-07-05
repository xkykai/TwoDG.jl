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

        # the recursive face element carries the former 1D tables (aliases)
        @test master.face isa ReferenceElement{1}
        @test master.sh1d === master.face.shap
        @test master.ma1d === master.face.mass
        @test master.gw1d === master.face.gwgh
        @test master.ploc1d === master.face.plocal

        # the constructive symmetry-matched perm (used to build the 3D tables)
        # reproduces the hand-built 2D orderings
        @test TwoDG.Masters.build_face_perm(Val(2), master.plocal,
                                            master.face.plocal) == master.perm
    end

    @testset "3D collapsed Gauss-Jacobi quadrature (degree $d)" for d in (2, 5, 9)
        x, w = gaussquad3d(d)
        # ∫_T ξ^a η^b ζ^c dV = a! b! c! / (a+b+c+3)! on the unit tetrahedron
        exact(a, b, c) = Float64(factorial(big(a)) * factorial(big(b)) *
                                 factorial(big(c)) // factorial(big(a + b + c + 3)))
        @test all(abs(sum(w .* x[:, 1] .^ a .* x[:, 2] .^ b .* x[:, 3] .^ c) -
                      exact(a, b, c)) < 1e-13
                  for a in 0:d for b in 0:(d - a) for c in 0:(d - a - b))
        @test all(>(0), w)   # Gauss-Jacobi product weights are positive
    end

    @testset "3D Koornwinder (PKD) basis is orthonormal (p = $p)" for p in (2, 4)
        x, w = gaussquad3d(2p + 1)
        V, _, _, _ = koornwinder3d(x, p)
        @test norm(V' * Diagonal(w) * V - I) < 1e-12
    end

    @testset "3D PKD derivatives match finite differences (p = 3)" begin
        p = 3
        pts = [0.21 0.17 0.33; 0.05 0.1 0.6; 0.3 0.3 0.3]
        h = 1e-6
        _, fx, fy, fz = koornwinder3d(pts, p)
        for (d, fd) in enumerate((fx, fy, fz))
            dx = zeros(1, 3)
            dx[d] = h
            fp, _, _, _ = koornwinder3d(pts .+ dx, p)
            fm, _, _, _ = koornwinder3d(pts .- dx, p)
            @test maximum(abs, (fp - fm) / 2h - fd) < 1e-7
        end
    end

    @testset "tetrahedral reference element (p = $p)" for p in (1, 2, 3)
        master = ReferenceElement(p; dim=3)
        npl = (p + 1) * (p + 2) * (p + 3) ÷ 6
        npf = (p + 1) * (p + 2) ÷ 2

        # recursive structure: tet -> triangle -> segment
        @test master isa ReferenceElement{3}
        @test master.face isa ReferenceElement{2}
        @test master.face.face isa ReferenceElement{1}
        @test size(master.plocal) == (npl, 4)
        @test size(master.perm) == (npf, 4, 6)

        # partition of unity and zero derivative sums at all quadrature points
        @test maximum(abs, sum(master.shap[:, 1, :], dims=1) .- 1) < 1e-11
        @test maximum(abs, sum(master.shap[:, 2:4, :], dims=1)) < 1e-9

        # reference mass matrix: SPD, ∫_T 1 dV = 1/6
        @test norm(master.mass - master.mass') < 1e-14
        @test isposdef(Symmetric(master.mass))
        @test abs(sum(master.mass) - 1 / 6) < 1e-12

        # node-set symmetry (D7): every perm column is a permutation of the
        # face's node set, and lies on its face
        @test all(all(master.plocal[master.perm[:, k, 1], k] .< 1e-6) for k in 1:4)
        @test all(allunique(master.perm[:, s, o]) &&
                  sort(master.perm[:, s, o]) == sort(master.perm[:, s, 1])
                  for s in 1:4, o in 1:6)

        # face compatibility: the face restriction of the volume nodes equals
        # the triangle face element's node set (in face barycentric coords)
        for s in 1:4
            fv = TwoDG.Masters.face_vertices(Val(3), s)
            facebary = master.plocal[master.perm[:, s, 1], collect(fv)]
            @test maximum(abs, facebary - master.face.plocal) < 1e-12
        end

        # the 6 orientation permutations form the symmetry group of the
        # triangle (closure under composition)
        ops = TwoDG.Masters.orientation_permutations(Val(3))
        compose(a, b) = ntuple(i -> a[b[i]], 3)
        @test all(compose(a, b) in ops for a in ops, b in ops)
    end
end
