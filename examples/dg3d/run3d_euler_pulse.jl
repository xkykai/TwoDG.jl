# 3D acoustics with the compressible Euler equations: a Gaussian pressure
# pulse released at the center of a closed box with slip walls. The pulse
# expands as a spherical acoustic wave, reflects off the six walls, and
# refocuses — a classic "pretty" case for volume rendering / isosurfaces.
# Writes a numbered .vtu time series that ParaView groups automatically.
#
#   julia --project=. examples/dg3d/run3d_euler_pulse.jl

using TwoDG
using StaticArrays
using Printf

γ = 1.4
p_pulse(x, y, z) = 1.0 + 0.2 * exp(-((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2) / 0.01)
ics = [(x, y, z) -> 1.0,                                # ρ (uniform)
       (x, y, z) -> 0.0, (x, y, z) -> 0.0, (x, y, z) -> 0.0,
       (x, y, z) -> p_pulse(x, y, z) / (γ - 1)]         # ρE (fluid at rest)

porder = 2
mesh = mkmesh_box(9, 9, 9, porder)
eq = EulerEquations{3}(γ=γ)
prob = DGProblem(eq, mesh; bc=ntuple(_ -> SlipWall(), 6), u0=ics)

dt = compute_dt(prob; cfl=0.3)
tfinal = 0.4                                            # ~ one wall reflection
nsnap = 10                                              # VTK frames
steps_per_snap = ceil(Int, tfinal / dt / nsnap)
dt = tfinal / (nsnap * steps_per_snap)
@info @sprintf("%d tets, dt = %.2e, %d steps, %d frames",
               size(mesh.t, 1), dt, nsnap * steps_per_snap, nsnap)

havevtk = Base.find_package("WriteVTK") !== nothing
if havevtk
    using WriteVTK
end
outdir = joinpath(@__DIR__, "output")
havevtk && mkpath(outdir)

# per-snapshot callback: dump pressure every steps_per_snap steps
frame = Ref(0)
function snapshot(state)
    state.step % steps_per_snap == 0 || return false
    frame[] += 1
    if havevtk
        pr = derived_field(pressure, eq, Array(state.u))
        save_vtk(mesh, pr, joinpath(outdir, @sprintf("pulse_%04d", frame[]));
                 names=(:p,))
    end
    @info @sprintf("t = %.3f (frame %d/%d)", state.t, frame[], nsnap)
    return false
end

sol = solve(prob, RK4(); dt, tfinal, callback=snapshot)

pr = derived_field(pressure, eq, sol.u)
@info @sprintf("final pressure range: [%.4f, %.4f]", extrema(pr)...)
havevtk || @info "install/load WriteVTK.jl to write the ParaView time series"
