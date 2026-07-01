# Mesh generators: connectivity and geometry invariants shared by every mesh.

using TwoDG
using Test

"Structural invariants every mesh must satisfy (f/t2f consistency, orientation)."
function check_mesh_invariants(mesh)
    (; p, t, f, t2f) = mesh
    nf, nt = size(f, 1), size(t, 1)
    bnd = f[:, 4] .< 0
    ni = count(!, bnd)

    # interior faces first, boundary faces last
    @test !any(bnd[1:ni]) && all(bnd[(ni + 1):end])
    # handshake: every element has 3 faces
    @test 3nt == 2ni + (nf - ni)

    # each face's endpoints are vertices of its adjacent element(s)
    @test all(issubset(f[i, 1:2], t[f[i, 3], :]) for i in 1:nf)
    @test all(issubset(f[i, 1:2], t[f[i, 4], :]) for i in 1:ni)

    # t2f and f agree, and every face is referenced the right number of times
    refcount = zeros(Int, nf)
    ok = true
    for el in 1:nt, k in 1:3
        fc = abs(t2f[el, k])
        refcount[fc] += 1
        ok &= (f[fc, 3] == el || f[fc, 4] == el)
    end
    @test ok
    @test all(refcount[1:ni] .== 2) && all(refcount[(ni + 1):end] .== 1)

    # positively oriented elements (counterclockwise vertices)
    signed_area(el) = (p[t[el, 2], 1] - p[t[el, 1], 1]) * (p[t[el, 3], 2] - p[t[el, 1], 2]) -
                      (p[t[el, 3], 1] - p[t[el, 1], 1]) * (p[t[el, 2], 2] - p[t[el, 1], 2])
    @test all(signed_area(el) > 0 for el in 1:nt)

    # straight elements: dgnodes are the barycentric image of the vertices
    straight = findall(!, mesh.tcurved)
    @test all(isapprox(mesh.dgnodes[:, :, el], mesh.plocal[:, 1:3] * p[t[el, :], :];
                       atol=1e-12) for el in straight)
end

@testset "Meshes" begin
    @testset "$name" for (name, mesh) in (
            ("square p2 parity 0", mkmesh_square(5, 4, 2, 0, 1)),
            ("square p3 parity 1", mkmesh_square(4, 5, 3, 1, 0)),
            ("Trefftz (curved)", mkmesh_trefftz(6, 12, 3)),
            ("L-shape", mkmesh_lshape(3, 2)))
        check_mesh_invariants(mesh)
    end

    @testset "uniform refinement" begin
        p, t = make_square_mesh(4, 4, 0)
        p2, t2 = uniref(p, t, 1)
        area(p, t) = sum((p[t[i, 2], 1] - p[t[i, 1], 1]) * (p[t[i, 3], 2] - p[t[i, 1], 2]) -
                         (p[t[i, 3], 1] - p[t[i, 1], 1]) * (p[t[i, 2], 2] - p[t[i, 1], 2])
                         for i in axes(t, 1)) / 2
        @test size(t2, 1) == 4 * size(t, 1)
        @test area(p2, t2) ≈ area(p, t) atol=1e-12
    end
end
