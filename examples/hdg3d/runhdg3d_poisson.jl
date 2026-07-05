# HDG on tetrahedra (THREED_PLAN Phase E1): Poisson on the unit box through
# the user-facing HDGProblem/solve API, with the p+2 superconvergent
# postprocessing — the acceptance property of the 3D HDG port.
#
#   -Δu = 3π² sin(πx) sin(πy) sin(πz),  u = 0 on ∂Ω,
#
# has the exact solution u = sin(πx) sin(πy) sin(πz). The observed rates
# should approach p+1 for u and p+2 for the postprocessed u*.
# Writes ParaView output when WriteVTK is loaded.
#
#   julia --project=. examples/hdg3d/runhdg3d_poisson.jl

using TwoDG
using Printf

exact(x, y, z) = sin(π * x) * sin(π * y) * sin(π * z)
source(p) = reshape(3π^2 .* sin.(π .* p[:, 1]) .* sin.(π .* p[:, 2]) .*
                    sin.(π .* p[:, 3]), :, 1)

porder = 2
ngauss = 4 * (porder + 1)
param = Dict(:kappa => 1.0, :c => zeros(3), :taud => 1.0)
dbc(p) = zeros(size(p, 1), 1)

errs_u, errs_ustar = Float64[], Float64[]
grids = [3, 5, 9]
local mesh, sol, ustar, mesh1
for n in grids
    # user-facing solve (batched assembly + sparse-direct trace solve)
    global mesh = mkmesh_box(n, n, n, porder)
    prob = HDGProblem(PoissonEquation{3}(), mesh; bc=Dirichlet(0.0), source=source)
    global sol = solve(prob, Direct())

    # superconvergent postprocessing needs the p+1 mesh/master pair built
    # with the same quadrature rule
    master = ReferenceElement(mesh, ngauss)
    global mesh1 = mkmesh_box(n, n, n, porder + 1)
    master1 = ReferenceElement(mesh1, ngauss)
    global ustar = hdg_postprocess(master, mesh, master1, mesh1, sol.u, sol.q)

    push!(errs_u, l2error(sol, exact))
    push!(errs_ustar, l2error(mesh1, ustar[:, 1, :], exact))
    @info @sprintf("n = %d (%d tets): |u - uₕ| = %.3e   |u - u*| = %.3e",
                   n, size(mesh.t, 1), errs_u[end], errs_ustar[end])
end
for i in 2:length(errs_u)
    @info @sprintf("rates %d -> %d:  u %.2f (design %d)   u* %.2f (design %d)",
                   grids[i-1], grids[i],
                   log2(errs_u[i-1] / errs_u[i]), porder + 1,
                   log2(errs_ustar[i-1] / errs_ustar[i]), porder + 2)
end

# ParaView output (optional): load WriteVTK to enable save_vtk
if Base.find_package("WriteVTK") !== nothing
    using WriteVTK
    outdir = joinpath(@__DIR__, "output")
    mkpath(outdir)
    files = save_vtk(mesh1, ustar[:, 1, :], joinpath(outdir, "hdg_poisson_ustar");
                     names=(:ustar,))
    @info "wrote $(files[1]) — open in ParaView"
else
    @info "install/load WriteVTK.jl to write ParaView output"
end
