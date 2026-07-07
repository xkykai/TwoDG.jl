# Solvers

Three discretizations share the same meshes, equations, and boundary
conditions, behind one CommonSolve `solve`:

| Problem | Method | Algorithms |
|---|---|---|
| [`DGProblem`](@ref) | explicit (L)DG | [`RK4`](@ref), or any OrdinaryDiffEq stepper via [`semidiscretize`](@ref) |
| [`HDGProblem`](@ref) | implicit HDG (static condensation) | [`GMRES`](@ref) (default), [`Direct`](@ref) |
| [`CGProblem`](@ref) | continuous Galerkin | [`Direct`](@ref) (default), [`ConjugateGradient`](@ref), [`GMRES`](@ref) |

## Explicit DG / LDG

`solve(prob::DGProblem, RK4(); dt, tfinal)` advances the semidiscrete system
with a low-storage four-stage Runge–Kutta loop built from
KernelAbstractions kernels — the residual ([`inviscid_residual!`](@ref) for
hyperbolic systems, [`viscous_residual!`](@ref) with LDG viscous fluxes for
convection–diffusion) never allocates, and `ArrayT = CuArray` moves the
whole loop to a GPU. [`compute_dt`](@ref) picks a CFL-limited step from the
mesh, the polynomial order, and the equation's wave speed/diffusivity.

The `callback` keyword runs user code between steps — from a bare closure
`state -> ...` (return `true` to stop early) to a [`CallbackSet`](@ref) of
schedules, analysis, snapshot, and checkpoint callbacks; see
[Callbacks and diagnostics](callbacks.md).

For adaptive or specialized time integration, [`semidiscretize`](@ref)
returns a SciML `ODEProblem` wrapping the same residual.

## HDG

Hybridizable DG condenses each element onto its face traces: the globally
coupled system involves only `porder + 1` unknowns per face, and everything
element-interior is recovered locally afterwards. `solve(prob::HDGProblem)`
uses the batched path by default — batched local solves and assembly
([`HDGBatch`](@ref)), a block-Jacobi-preconditioned restarted GMRES on the
trace system ([`HDGSystem`](@ref), Krylov.jl), and batched recovery — all of
it KA-portable (`ArrayT = CuArray`). [`Direct`](@ref) assembles the same
trace system and factorizes it sparsely on the CPU (the robust/debugging
option).

Two HDG-specific accuracy notes, learned the hard way:

- **Superconvergence needs consistent data.** [`hdg_postprocess`](@ref)
  recovers a `p+1` field converging at order `p+2`, but only when the source
  is supplied at quadrature points and Dirichlet data is L2-projected onto
  the trace space — pointwise-interpolated boundary data destroys the extra
  order.
- **Superconvergence needs one discrete geometry.** On curved meshes the
  `p+1` mesh must carry the *same* isoparametric map as the `p` mesh — call
  [`match_geometry!`](@ref)`(master, mesh, master1, mesh1)` before
  postprocessing. Independently projected boundary nodes give two maps that
  differ by `O(h^{p+1})`, and `u*` gains nothing over `u`.
- The stabilization parameter `τ` (`stabilization` keyword) trades
  robustness for accuracy; `1.0` is a good default for diffusion-dominated
  problems.

The incompressible Navier–Stokes solvers ([`hdg_ns_solve`](@ref),
[`hdg_ns_step`](@ref), with [`hdg_ns_postprocess`](@ref) for the
exactly divergence-free postprocessed velocity) are driver-level APIs and
work on triangles and tetrahedra alike (the divergence-free postprocessing
is 2D-only for now) — see `examples/hdg/runhdg_ns_kovasznay.jl` (steady
verification), `runhdg_ns_boussinesq.jl` (Boussinesq natural convection),
and `examples/hdg3d/runhdg3d_ns_beltrami.jl` (3D Beltrami flow).

## CG

`solve(prob::CGProblem)` assembles the global sparse stiffness matrix from
batched element matrices and factorizes it directly — Cholesky when the
operator is SPD (no convection), LU otherwise. The iterative algorithms are
matrix-free: one fused gather → element-matvec → scatter kernel per operator
application, Jacobi-preconditioned Krylov iterations
([`ConjugateGradient`](@ref) for SPD problems, [`GMRES`](@ref) with
convection), running on CPU or GPU via `ArrayT`. The low-level entry points
are [`cg_solve`](@ref) (direct) and [`cg_parsolve`](@ref) (iterative). Both
work on triangles and tetrahedra — `CGProblem(PoissonEquation{3}(), mesh;
source)` on a tetrahedral mesh solves 3D Poisson at the design rate.

## Solutions

Every `solve` returns a solution object carrying the field(s), the problem
(mesh included), and solver metadata: `sol.u` is always `(npl, nc, nt)`;
HDG solutions add the flux `q` and trace `uhat`; iterative solves record
`iterations`. [`l2error`](@ref)`(sol, exact)` computes grid-converged L2
errors, and `scaplot(sol.prob.mesh, sol.u[:, 1, :])` plots a component once
a Makie backend is loaded.
