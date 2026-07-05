# TwoDGWriteVTKExt: high-order Lagrange VTK output (THREED_PLAN C6/D10).
# Self-skips unless WriteVTK.jl is available (it is a test dependency, so
# `Pkg.test()` always runs it; direct `include` runs skip gracefully).

using TwoDG
using Test

if Base.find_package("WriteVTK") === nothing
    @info "Skipping WriteVTK extension test (WriteVTK.jl not installed)"
else
    using WriteVTK

    @testset "TwoDGWriteVTKExt (save_vtk)" begin
        ext = Base.get_extension(TwoDG, :TwoDGWriteVTKExt)
        @test ext !== nothing

        @testset "VTK Lagrange lattice invariants (p = $p)" for p in 1:4
            # triangle: vertices first, then points on edge 12 / 23 / 31
            tri = ext.vtk_triangle_lattice(p)
            @test length(tri) == (p + 1) * (p + 2) ÷ 2
            @test allunique(tri)
            @test tri[1:3] == [(p, 0, 0), (0, p, 0), (0, 0, p)]
            for a in 1:(p - 1)
                @test tri[3 + a][3] == 0                       # edge 1→2
                @test tri[3 + (p - 1) + a][1] == 0             # edge 2→3
                @test tri[3 + 2(p - 1) + a][2] == 0            # edge 3→1
            end

            # tetrahedron: vertices, edge points on their edges, face points
            # off exactly the opposite vertex, interior strictly inside
            tet = ext.vtk_tet_lattice(p)
            @test length(tet) == (p + 1) * (p + 2) * (p + 3) ÷ 6
            @test allunique(tet)
            @test tet[1:4] == [(p, 0, 0, 0), (0, p, 0, 0), (0, 0, p, 0), (0, 0, 0, p)]
            k = 4
            for (a, b) in ext.TET_EDGES, s in 1:(p - 1)
                k += 1
                q = tet[k]
                @test q[a] == p - s && q[b] == s && sum(q) == p
            end
            nfp = p >= 3 ? (p - 2) * (p - 1) ÷ 2 : 0
            for (m, fverts) in enumerate(ext.TET_FACES), j in 1:nfp
                q = tet[4 + 6 * (p - 1) + (m - 1) * nfp + j]
                opp = only(setdiff(1:4, fverts))
                @test q[opp] == 0 && all(q[v] >= 1 for v in fverts)
            end
            for q in tet[(4 + 6 * (p - 1) + 4 * nfp + 1):end]
                @test all(>=(1), q)
            end
        end

        @testset "2D and 3D output round-trip" begin
            mktempdir() do dir
                # 2D curved mesh, system field with names from the equation
                mesh2 = mkmesh_trefftz(4, 8, 3)
                u2 = initu(mesh2, 2, [(x, y) -> x + y, (x, y) -> x * y])
                files = save_vtk(mesh2, u2, joinpath(dir, "trefftz");
                                 names=(:a, :b))
                @test all(isfile, files)

                # 3D curved octant through the solution-object path
                mesh3 = mkmesh_box(3, 3, 3, 2)
                prob = DGProblem(ConvectionEquation([1.0, 0.0, 0.0]), mesh3;
                                 bc=ntuple(_ -> Dirichlet(0.0), 6),
                                 u0=[(x, y, z) -> sin(π * x) * y * z])
                sol = solve(prob, RK4(); dt=1e-3, nstep=2)
                files3 = save_vtk(sol, joinpath(dir, "box3d"))
                @test all(isfile, files3)

                # sanity: the written points reproduce the mesh volume nodes
                # (scalar matrix form + default names)
                files1 = save_vtk(mesh3, mesh3.dgnodes[:, 1, :], joinpath(dir, "coords"))
                @test all(isfile, files1)
            end
        end
    end
end
