# HDG incompressible Navier-Stokes on tetrahedra (THREED_PLAN Phase E3):
# steady Beltrami flow on the unit box, solved by Newton iteration on the
# batched assembly path with cache reuse.
#
# The Ethier-Steinman Beltrami field (IJNMF 19, 1994) has vorticity parallel
# to velocity, so the convection term is a pure gradient, (u·∇)u = ∇(|u|²/2),
# absorbed by the pressure p = -|u|²/2. Because Δu_B = -d² u_B, the field
# solves the *steady* Navier-Stokes equations with the body force
#
#     f = -ν Δu_B = ν d² u_B,
#
# giving a genuinely three-dimensional exact solution that exercises all
# velocity components, all nine gradient entries, and the full Newton
# linearization. The observed velocity rate should approach p+1.
# Writes ParaView output when WriteVTK is loaded.
#
#   julia --project=. examples/hdg3d/runhdg3d_ns_beltrami.jl

using TwoDG
using LinearAlgebra
using Printf

a, d = π / 4, π / 2
ν = 1.0
uB(x, y, z) = [-a * (exp(a * x) * sin(a * y + d * z) + exp(a * z) * cos(a * x + d * y)),
               -a * (exp(a * y) * sin(a * z + d * x) + exp(a * x) * cos(a * y + d * z)),
               -a * (exp(a * z) * sin(a * x + d * y) + exp(a * y) * cos(a * z + d * x))]
fB(p) = ν * d^2 .* uB(p[1], p[2], p[3])
dbc(p) = uB(p[1], p[2], p[3])

porder = 2
τ = 2.0                       # τ ≈ ν/ℓ + |u| for the Beltrami velocity scale

"Newton iteration on the batched step, reusing the factorization cache."
function solve_beltrami(mesh, master; maxiter=10, tol=1e-10)
    u = Λ = cache = nothing
    res = nothing
    for iter in 1:maxiter
        res = hdg_ns_step_batched(master, mesh, ν, dbc; τ, source=fB, u, Λ, cache)
        Δ = Λ === nothing ? Inf : norm(res.Λ .- Λ) / max(norm(res.Λ), eps())
        @info @sprintf("  Newton %d: Δλ = %.2e", iter, Δ)
        u, Λ, cache = res.u, res.Λ, res.cache
        Δ < tol && break
    end
    return res
end

errs = Float64[]
grids = [3, 5]
local mesh, res
for n in grids
    global mesh = mkmesh_box(n, n, n, porder)
    master = ReferenceElement(mesh, 3 * (porder + 1))
    @info "solving on $(size(mesh.t, 1)) tets"
    global res = solve_beltrami(mesh, master)
    err = sqrt(sum(abs2, [l2error(mesh, res.u[:, c, :],
                                  (x, y, z) -> uB(x, y, z)[c]) for c in 1:3]))
    push!(errs, err)
    @info @sprintf("n = %d: |u - uₕ| = %.4e   max |tr ∇u| = %.2e", n, err,
                   maximum(abs.(res.gradu[:, 1, :] .+ res.gradu[:, 5, :] .+
                                res.gradu[:, 9, :])))
end
@info @sprintf("velocity rate %d -> %d: %.2f (design %d)", grids[1], grids[2],
               log2(errs[1] / errs[2]), porder + 1)

if Base.find_package("WriteVTK") !== nothing
    using WriteVTK
    outdir = joinpath(@__DIR__, "output")
    mkpath(outdir)
    fields = cat(res.u, reshape(res.p, size(res.p, 1), 1, :); dims=2)
    files = save_vtk(mesh, fields, joinpath(outdir, "beltrami");
                     names=(:u, :v, :w, :p))
    @info "wrote $(files[1])"
else
    @info "install/load WriteVTK.jl to write ParaView output"
end
