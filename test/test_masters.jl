# Master element: quadrature rules, Koornwinder bases, nodal shape functions.

using TwoDG
using Test
using LinearAlgebra
import NodesAndModes

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

    # d ≤ 10 exercises the tabulated symmetric (Witherden-Vincent) rules,
    # d = 12 the collapsed Gauss-Jacobi product fallback
    @testset "3D tet quadrature (degree $d)" for d in (2, 4, 5, 9, 10, 12)
        x, w = gaussquad3d(d)
        # ∫_T ξ^a η^b ζ^c dV = a! b! c! / (a+b+c+3)! on the unit tetrahedron
        exact(a, b, c) = Float64(factorial(big(a)) * factorial(big(b)) *
                                 factorial(big(c)) // factorial(big(a + b + c + 3)))
        @test all(abs(sum(w .* x[:, 1] .^ a .* x[:, 2] .^ b .* x[:, 3] .^ c) -
                      exact(a, b, c)) < 1e-13
                  for a in 0:d for b in 0:(d - a) for c in 0:(d - a - b))
        @test all(>(0), w)                       # positive weights
        @test all(>(0), x) && all(<(1), sum(x; dims=2))   # interior points
        d <= 10 && @test size(x, 1) < ceil(Int, (d + 1) / 2)^3   # fewer than product rule
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

        # node-set symmetry: every perm column is a permutation of the
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

    @testset "NodesAndModes.jl oracle (p = $p)" for p in (2, 4)
        # Independent cross-validation of the in-house reference-element code
        # against NodesAndModes.jl. Conventions differ — NodesAndModes uses the
        # bi-unit simplex (vertices at ±1), TwoDG the unit simplex — so nodes
        # map by x = (r + 1)/2 and an orthonormal basis picks up the volume
        # ratio: on the tet dr = 8 dx, so ψ(x) = √8 φ_NM(2x - 1) is orthonormal
        # on the unit tet (factor 2 on the triangle).
        #
        # The cross-Gram C_ij = ∫ φ_i ψ_j over the unit simplex, evaluated with
        # TwoDG's quadrature, must be orthogonal (C Cᵀ = I): both bases are
        # orthonormal and span the same polynomial space, and the rule is exact
        # at degree 2p. This validates basis normalization, completeness, AND
        # quadrature in one shot, against an implementation we don't own.

        # --- tetrahedron ---
        gp, gw = gaussquad3d(2p)
        r, s, t = (2 .* gp[:, i] .- 1 for i in 1:3)
        Vnm = NodesAndModes.basis(NodesAndModes.Tet(), p, r, s, t)[1] .* sqrt(8)
        Vtd = koornwinder3d(gp, p)[1]
        C = Vtd' * (gw .* Vnm)
        @test norm(C * C' - I) < 1e-10
        @test size(C, 1) == (p + 1) * (p + 2) * (p + 3) ÷ 6

        # equispaced node sets coincide as point sets
        re, se, te = NodesAndModes.equi_nodes(NodesAndModes.Tet(), p)
        nm_nodes = sortslices([re se te] ./ 2 .+ 0.5; dims=1)
        td_nodes = sortslices(localpnts3d(p)[:, 2:4]; dims=1)
        @test maximum(abs, nm_nodes - td_nodes) < 1e-12

        # --- triangle (the face element of the tet) ---
        gp2, gw2 = gaussquad2d(2p)
        r2, s2 = (2 .* gp2[:, i] .- 1 for i in 1:2)
        Vnm2 = NodesAndModes.basis(NodesAndModes.Tri(), p, r2, s2)[1] .* 2
        Vtd2 = koornwinder2d(gp2, p)[1]
        C2 = Vtd2' * (gw2 .* Vnm2)
        @test norm(C2 * C2' - I) < 1e-10
    end

    @testset "3D warp-and-blend nodes (nodetype = 1)" begin
        # set distance (roundoff noise makes row-sorting comparisons unusable)
        setdist(A, B) = maximum(minimum(sqrt(sum(abs2, A[i, :] .- B[j, :]))
                                        for j in axes(B, 1)) for i in axes(A, 1))

        # p ≤ 3: the warp is uniquely determined and must match the
        # NodesAndModes.jl warp-and-blend nodes exactly
        for p in (2, 3)
            nm = hcat(NodesAndModes.nodes(NodesAndModes.Tet(), p)...) ./ 2 .+ 0.5
            @test setdist(nm, localpnts3d(p, 1)[:, 2:4]) < 1e-12
        end

        # higher p: NodesAndModes uses the interpolatory variant so the sets
        # differ; assert the properties that matter instead — Vandermonde
        # conditioning beats the uniform lattice, and the set satisfies the
        # symmetry-group/face-restriction requirements (the constructor
        # asserts both while building perm)
        for p in (6, 8)
            Vu, = koornwinder3d(localpnts3d(p, 0)[:, 2:4], p)
            plw = localpnts3d(p, 1)
            Vw, = koornwinder3d(plw[:, 2:4], p)
            @test cond(Vw) < 0.85 * cond(Vu)   # measured 26 vs 33 (p=6), 56 vs 92 (p=8)
            master = ReferenceElement(plw, p; pgauss=2p)
            @test master isa ReferenceElement{3}
        end
    end
end
