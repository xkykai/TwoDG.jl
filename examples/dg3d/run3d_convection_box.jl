# Entry-level 3D example: linear convection of a smooth profile through the
# unit box, solved with explicit DG on tetrahedra (Kuhn 6-tet cells).
# u_t + ∇·(v u) = 0 with constant v transports the profile unchanged, so the
# exact solution is known and the observed convergence rate can be checked
# against the design rate p+1. Writes ParaView output when WriteVTK is loaded.
#
#   julia --project=. examples/dg3d/run3d_convection_box.jl

using TwoDG
using StaticArrays
using Printf

v = SVector(1.0, 0.5, 0.25)                     # constant transport velocity
u0(x, y, z) = sin(π * x) * sin(π * y) * sin(π * z)
uexact(x, t) = u0((x .- v .* t)...)

porder = 2
tfinal = 0.2
dt = 1e-3

errs = Float64[]
grids = [3, 5, 9]
local mesh, u
for nx in grids
    global mesh = mkmesh_box(nx, nx, nx, porder)
    master = ReferenceElement(mesh)
    ctx = DGContext(master, mesh)

    # time-dependent Dirichlet inflow data = the exact solution
    phys = DGPhysics(ConvectionEquation(collect(v));
                     boundary_conditions=ntuple(_ -> Dirichlet(uexact), 6))

    global u = initu(mesh, 1, [u0])
    rk4_ka!(ctx, phys, u, 0.0, dt, round(Int, tfinal / dt))

    err = l2error(mesh, u[:, 1, :], (x, y, z) -> uexact(SVector(x, y, z), tfinal))
    push!(errs, err)
    @info @sprintf("nx = %d (%d tets): L2 error = %.3e", nx, size(mesh.t, 1), err)
end
for i in 2:length(errs)
    @info @sprintf("rate %d -> %d: %.2f  (design rate %d)",
                   grids[i-1], grids[i], log2(errs[i-1] / errs[i]), porder + 1)
end

# ParaView output (optional): load WriteVTK to enable save_vtk
if Base.find_package("WriteVTK") !== nothing
    using WriteVTK
    outdir = joinpath(@__DIR__, "output")
    mkpath(outdir)
    files = save_vtk(mesh, u, joinpath(outdir, "convection_box"); names=(:u,))
    @info "wrote $(files[1]) — open in ParaView"
else
    @info "install/load WriteVTK.jl to write ParaView output"
end
