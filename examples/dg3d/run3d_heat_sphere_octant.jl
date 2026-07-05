# Curved 3D geometry: heat conduction in an octant of the unit ball, solved
# with LDG on curved (isoparametric) tetrahedra. The initial condition is the
# fundamental radial mode j₀(πr) = sin(πr)/(πr), which vanishes on the sphere
# and is symmetric across the coordinate planes, so with Dirichlet(0) on the
# curved sphere patch and insulated (Neumann) symmetry planes the exact
# solution of the full ball restricts to the octant:
#
#     u(r, t) = j₀(πr) exp(-κ π² t)
#
# This exercises the whole curved-geometry pipeline: Bey `uniref` of a corner
# tet, radial vertex projection, curved-face node projection in `discretize`,
# and the isoparametric metric — with a known solution to measure against.
#
#   julia --project=. examples/dg3d/run3d_heat_sphere_octant.jl

using TwoDG
using LinearAlgebra
using StaticArrays
using Printf

κ = 0.05
j0(x) = x == 0 ? 1.0 : sin(x) / x
u0(x, y, z) = j0(π * hypot(x, y, z))
uexact(x, y, z, t) = u0(x, y, z) * exp(-κ * π^2 * t)

# --- octant mesh: unit corner tet, red-refined, hypotenuse plane projected
# radially onto the sphere so the linear mesh respects the boundary
nref = 2
p0 = [0.0 0 0; 1.0 0 0; 0 1.0 0; 0 0 1.0]
t0 = reshape([1, 2, 3, 4], 1, 4)
p1, t1 = uniref(p0, t0, nref)
for i in axes(p1, 1)
    if abs(sum(p1[i, :]) - 1) < 1e-12
        p1[i, :] ./= norm(p1[i, :])
    end
end

ϵ = 1e-6
geo = MeshGeometry(p1, t1;
                   boundaries=(sphere=p -> vec(sqrt.(sum(abs2, p; dims=2)) .> 0.8),
                               xy=p -> p[:, 3] .< ϵ,
                               xz=p -> p[:, 2] .< ϵ,
                               yz=p -> p[:, 1] .< ϵ),
                   curved=[:sphere],
                   fd=(x -> sqrt(x[1]^2 + x[2]^2 + x[3]^2) - 1,
                       x -> x[3], x -> x[2], x -> x[1]))

porder = 3
mesh = discretize(geo, porder)
master = ReferenceElement(mesh)
ctx = DGContext(master, mesh)
@info @sprintf("octant mesh: %d curved tets of %d, volume error %.1e",
               count(mesh.tcurved), size(mesh.t, 1), abs(sum(ctx.wjac) - π / 6))

# --- LDG heat equation: Dirichlet(0) on the sphere, insulated symmetry planes
phys = DGPhysics(ConvectionDiffusionEquation(SVector(0.0, 0.0, 0.0), κ);
                 boundary_conditions=(Dirichlet(0.0), Neumann(), Neumann(), Neumann()),
                 stabilization=LDGStabilization(10.0, 0.0))

dt, nstep = 2e-4, 500
tfinal = dt * nstep
u = initu(mesh, 1, [u0])
rk4_ka!(rldgexpl!, ctx, phys, u, 0.0, dt, nstep)

err = l2error(mesh, u[:, 1, :], (x, y, z) -> uexact(x, y, z, tfinal))
@info @sprintf("t = %.2f: L2 error vs exact radial mode = %.3e (p = %d, %d tets)",
               tfinal, err, porder, size(mesh.t, 1))

if Base.find_package("WriteVTK") !== nothing
    using WriteVTK
    outdir = joinpath(@__DIR__, "output")
    mkpath(outdir)
    files = save_vtk(mesh, u, joinpath(outdir, "heat_octant"); names=(:u,))
    @info "wrote $(files[1]) — the curved sphere patch renders exactly in ParaView"
else
    @info "install/load WriteVTK.jl to write ParaView output"
end
