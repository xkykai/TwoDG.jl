# Smoke test for the TwoDGGmshExt package extension (mkmesh_naca). Gmsh.jl
# is deliberately not a test dependency (it downloads the Gmsh SDK); this
# file self-skips unless Gmsh.jl is available in the active environment,
# e.g. `julia --project=test-env` with both TwoDG and Gmsh added.

using TwoDG
using Test

if Base.find_package("Gmsh") === nothing
    @info "Skipping Gmsh extension smoke test (Gmsh.jl not installed)"
else
    using Gmsh

    @testset "TwoDGGmshExt (mkmesh_naca)" begin
        mktempdir() do dir
            cd(dir) do   # gmsh_naca writes "<name>.msh" into the cwd
                porder = 2
                mesh = mkmesh_naca(10, porder)

                @test size(mesh.p, 2) == 2
                @test size(mesh.t, 2) == 3
                @test size(mesh.dgnodes, 1) == (porder + 1) * (porder + 2) ÷ 2
                @test size(mesh.dgnodes, 3) == size(mesh.t, 1)
                @test all(isfinite, mesh.dgnodes)
                @test boundary_names(mesh) == [:airfoil, :left, :right, :bottom, :top]

                # the airfoil boundary must exist, be curved, and its
                # high-order nodes must sit on the NACA surface
                bnd = mesh.f[:, 4] .< 0
                @test any(mesh.f[bnd, 4] .== -1)
                @test any(mesh.fcurved)

                naca_y(x, t=10) = 0.05 * t * (0.2969 * sqrt(abs(x)) - 0.1260 * x -
                                              0.3516 * x^2 + 0.2843 * x^3 - 0.1015 * x^4)
                airfoil_faces = findall(i -> mesh.f[i, 4] == -1, axes(mesh.f, 1))
                @test !isempty(airfoil_faces)
                # face endpoint vertices were projected onto the airfoil
                worst = maximum(airfoil_faces) do i
                    maximum(mesh.f[i, 1:2]) do vtx
                        x, y = mesh.p[vtx, 1], mesh.p[vtx, 2]
                        abs(abs(y) - naca_y(clamp(x, 0.0, 1.0)))
                    end
                end
                @test worst < 1e-2
            end
        end
    end
end
