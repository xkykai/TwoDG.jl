# 3D compressible Euler: isentropic vortex advected through the unit box —
# the classic exact-solution accuracy benchmark, run through the high-level
# DGProblem / solve / compute_dt API. The vortex axis is aligned with z and
# the free stream has w∞ ≠ 0, so all three flux directions and all six
# boundaries are exercised. Writes density and Mach number for ParaView.
#
#   julia --project=. examples/dg3d/run3d_euler_vortex.jl

using TwoDG
using StaticArrays
using Printf

γ = 1.4
β, R = 1.0, 0.25                    # vortex strength and core radius
v∞ = SVector(1.0, 0.5, 0.25)

# exact solution: the standard isentropic vortex, rescaled by R (spatial
# rescaling preserves exactness) and translated with the free stream
function conserved(x, y, z, t)
    x̂ = (x - 0.5 - v∞[1] * t) / R
    ŷ = (y - 0.5 - v∞[2] * t) / R
    f = exp((1 - x̂^2 - ŷ^2) / 2)
    Θ = 1 - (γ - 1) * β^2 / (8γ * π^2) * f^2            # temperature
    ρ = Θ^(1 / (γ - 1))
    vel = SVector(v∞[1] - β / (2π) * ŷ * f, v∞[2] + β / (2π) * x̂ * f, v∞[3])
    return SVector(ρ, ρ * vel[1], ρ * vel[2], ρ * vel[3],
                   Θ^(γ / (γ - 1)) / (γ - 1) + ρ * sum(abs2, vel) / 2)
end
uex(x, t) = conserved(x[1], x[2], x[3], t)

porder = 2
tfinal = 0.1
eq = EulerEquations{3}(γ=γ)

errs = Float64[]
grids = [3, 5, 9]
local sol, mesh
for nx in grids
    global mesh = mkmesh_box(nx, nx, nx, porder)
    prob = DGProblem(eq, mesh;
                     bc=ntuple(_ -> Dirichlet(uex), 6),   # exact time-dependent data
                     u0=[(x, y, z) -> conserved(x, y, z, 0.0)[c] for c in 1:5])

    dt = compute_dt(prob; cfl=0.3)                        # CFL-limited step
    nstep = ceil(Int, tfinal / dt)
    global sol = solve(prob, RK4(); dt=tfinal / nstep, tfinal)

    err = l2error(mesh, sol.u[:, 1, :], (x, y, z) -> conserved(x, y, z, tfinal)[1])
    push!(errs, err)
    @info @sprintf("nx = %d (%d tets): dt = %.2e, L2(ρ) error = %.3e",
                   nx, size(mesh.t, 1), dt, err)
end
for i in 2:length(errs)
    @info @sprintf("rate %d -> %d: %.2f  (design rate %d)",
                   grids[i-1], grids[i], log2(errs[i-1] / errs[i]), porder + 1)
end

if Base.find_package("WriteVTK") !== nothing
    using WriteVTK
    outdir = joinpath(@__DIR__, "output")
    mkpath(outdir)
    files = save_vtk(sol, joinpath(outdir, "euler_vortex"))      # ρ, ρu, ρv, ρw, ρE
    M = derived_field(mach, eq, sol.u)                           # pointwise Mach number
    files2 = save_vtk(mesh, M, joinpath(outdir, "euler_vortex_mach"); names=(:M,))
    @info "wrote $(files[1]) and $(files2[1]) — open in ParaView"
else
    @info "install/load WriteVTK.jl to write ParaView output"
end
