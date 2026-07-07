"""
High-level problem/solve API: thin orchestration over the validated low-level
solvers. Physics (equations, boundary conditions, numerical fluxes) lives in
`TwoDG.Equations` and is passed through to the kernels *unchanged* — there is
no translation layer.

```julia
using TwoDG

mesh = mkmesh_square(17, 17, 3, 0, 1)
eq   = EulerEquations(γ = 1.4)
prob = DGProblem(eq, mesh; bc = [FarField(uinf), SlipWall(), FarField(uinf), FarField(uinf)],
                 u0 = [ρ0, ρu0, ρv0, ρE0])
sol  = solve(prob, RK4(); dt = 1e-3, tfinal = 1.0)
err  = l2error(sol, exact)      # or scaplot(sol.prob.mesh, sol.u[:, 1, :])
```
"""
module Interface

import CommonSolve
using CommonSolve: solve
using StaticArrays
using Adapt: adapt
using LinearAlgebra: norm
using ..Equations
using ..Masters: Master
using ..Meshes: Mesh
using ..Utils: initu, interpolate
using ..Geometry: min_inscribed_diameter
using ..DiscontinuousGalerkin: DGContext, DGPhysics, rinvexpl!, rldgexpl!, rk4_ka!,
                               RinvWorkspace, RldgWorkspace, _default_ka_ws
using ..HybridizableDiscontinuousGalerkin: hdg_direct_batched, hdg_parsolve,
                                           hdg_parsolve_batched
using ..ContinuousGalerkin: cg_solve, cg_parsolve
using ..Callbacks: SolveState, initialize!, finish!, load_checkpoint
import ..ContinuousGalerkin: l2error

export solve, semidiscretize, compute_dt,
    DGProblem, HDGProblem, CGProblem,
    RK4, Direct, GMRES, ConjugateGradient

# ---------------------------------------------------------------- algorithms

"Classic four-stage Runge-Kutta time integration (the GPU-tight internal stepper)."
struct RK4 end

"Direct (sparse LU) solve of the condensed system."
struct Direct end

"""
    GMRES(; restart=80, tol=1e-6, maxit=2000, preconditioner=true, batched=true)

Restarted, block-Jacobi-preconditioned GMRES (Krylov.jl) on the HDG trace
system. `batched=true` uses the batched KA assembly/recovery path
(`hdg_parsolve_batched`); otherwise the per-element threaded assembly
(`hdg_parsolve`).
"""
Base.@kwdef struct GMRES
    restart::Int = 80
    tol::Float64 = 1e-6
    maxit::Int = 2000
    preconditioner::Bool = true
    batched::Bool = true
end

"""
    ConjugateGradient(; tol=1e-10, maxit=5000, preconditioner=true)

Jacobi-preconditioned conjugate-gradient solve of the (SPD) CG stiffness
system, matrix-free on any KA backend (`ArrayT=CuArray` runs the iteration
on the GPU). Only valid for [`CGProblem`](@ref)s without convection; use
[`GMRES`](@ref) otherwise.
"""
Base.@kwdef struct ConjugateGradient
    tol::Float64 = 1e-10
    maxit::Int = 5000
    preconditioner::Bool = true
end

# ------------------------------------------------------------------ problems

"""
    DGProblem(equation, mesh; bc, u0, source=nothing, numerical_flux=nothing,
              stabilization=nothing, T=Float64)

Explicit (L)DG semidiscretization of `equation` on `mesh`.

- `bc` — one `BoundaryCondition` per boundary tag (a vector/tuple in tag
  order, or a `NamedTuple` keyed by the mesh's boundary names).
- `u0` — initial condition: an `(npl, nc, nt)` array or a vector of `nc`
  constants / `(x, y) -> value` functions.
- `source` — `nothing` or a pointwise source `(u, x, t) -> SVector`.
- `numerical_flux` — any callable `(eq, uL, uR, n, x, t) -> SVector`;
  defaults to `default_numerical_flux(equation)`.
- `stabilization` — LDG penalty policy ([`LDGStabilization`](@ref)) for
  diffusive equations; defaults to `default_stabilization(equation)`.

Solve with [`RK4`](@ref):

    solve(prob, RK4(); dt, tfinal (or nstep), t0=0.0, ArrayT=Array)

`ArrayT=CuArray` (with CUDA.jl loaded) runs the whole time loop on the GPU.
"""
struct DGProblem{E <: AbstractEquation, M, B, U, S, NF, ST, T <: AbstractFloat}
    equation       :: E
    mesh           :: M
    bc             :: B
    u0             :: U
    source         :: S
    numerical_flux :: NF
    stabilization  :: ST
end

function DGProblem(equation::AbstractEquation, mesh; bc, u0, source=nothing,
                   numerical_flux=nothing, stabilization=nothing,
                   T::Type{<:AbstractFloat}=Float64)
    _check_dims(equation, mesh)
    return DGProblem{typeof(equation), typeof(mesh), typeof(bc), typeof(u0),
                     typeof(source), typeof(numerical_flux), typeof(stabilization),
                     T}(equation, mesh, bc, u0, source, numerical_flux, stabilization)
    end

# equations and meshes carry their space dimension as a type parameter; a
# mismatch is a user error caught at problem construction (D8: the user never
# spells Dim, so the mesh is the source of truth)
function _check_dims(equation::AbstractEquation, mesh)
    ndims(equation) == ndims(mesh) ||
        throw(ArgumentError("$(nameof(typeof(equation))) is $(ndims(equation))D but the mesh is $(ndims(mesh))D"))
    return nothing
end

Base.eltype(::DGProblem{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, T}) where {T} = T

"""
    HDGProblem(equation, mesh; bc, source=nothing, stabilization=1.0)

Steady convection-diffusion (or Poisson) problem discretized with HDG, on
triangular (2D) or tetrahedral (3D) meshes. `bc` is a single
[`Dirichlet`](@ref) applied on the whole boundary (constant, `(x, y) -> g`,
or `(x, y, z) -> g`); `source` is `nothing` or a function of the
quadrature-point coordinate matrix `p -> values`. Solve with
[`Direct`](@ref) (sparse LU) or [`GMRES`](@ref) (default).
"""
struct HDGProblem{E <: AbstractEquation, M, B, S, T}
    equation      :: E
    mesh          :: M
    bc            :: B
    source        :: S
    stabilization :: T
end

function HDGProblem(equation::AbstractEquation, mesh; bc, source=nothing, stabilization=1.0)
    _check_dims(equation, mesh)
    return HDGProblem(equation, mesh, bc, source, stabilization)
end

function hdg_param(eq::ConvectionDiffusionEquation, taud)
    eq.velocity isa Function &&
        throw(ArgumentError("HDGProblem requires a constant convective velocity"))
    return Dict(:kappa => eq.κ, :c => collect(Float64, eq.velocity), :taud => taud)
end
hdg_param(eq::PoissonEquation{Dim}, taud) where {Dim} =
    Dict(:kappa => eq.κ, :c => zeros(Dim), :taud => taud)

"""
    CGProblem(equation, mesh; source, reaction=0.0)

Continuous-Galerkin discretization of a Poisson / convection-diffusion(-
reaction) problem with homogeneous Dirichlet boundaries. `source` is a
function `(x, y) -> value`. Solve with [`Direct`](@ref) (the default).
"""
struct CGProblem{E <: AbstractEquation, M, S, T}
    equation :: E
    mesh     :: M
    source   :: S
    reaction :: T
end

function CGProblem(equation::AbstractEquation, mesh; source, reaction=0.0)
    _check_dims(equation, mesh)
    return CGProblem(equation, mesh, source, reaction)
end

cg_param(eq::PoissonEquation{Dim}, s) where {Dim} = (; κ=eq.κ, c=zeros(Dim), s)
cg_param(eq::ConvectionDiffusionEquation, s) =
    (; κ=eq.κ, c=collect(Float64, eq.velocity), s)

# ------------------------------------------------------------------ solutions

"""
Solution of a [`DGProblem`](@ref): `u (npl, nc, nt)` at time `t`. The
`callbacks` field carries whatever was passed as `callback` to `solve`
(`nothing` by default), so callback histories — e.g. an
[`AnalysisCallback`](@ref)'s `time`/`data` — ride along on the solution.
"""
struct DGSolution{U, T, P, C}
    u         :: U
    t         :: T
    prob      :: P
    callbacks :: C
end
DGSolution(u, t, prob) = DGSolution(u, t, prob, nothing)

"""
Solution of an [`HDGProblem`](@ref): `u (npl, 1, nt)`, flux `q (npl, Dim, nt)`,
trace `uhat (nps, nf)` and GMRES iteration count (`0` for a direct solve).
"""
struct HDGSolution{U, Q, H, P}
    u          :: U
    q          :: Q
    uhat       :: H
    iterations :: Int
    prob       :: P
end

"""
Solution of a [`CGProblem`](@ref): `u (npl, 1, nt)` (DG-numbered for
plotting), the discrete `energy`, and the Krylov iteration count (`0` for a
direct solve).
"""
struct CGSolution{U, E, P}
    u          :: U
    energy     :: E
    iterations :: Int
    prob       :: P
end

const AnySolution = Union{DGSolution, HDGSolution, CGSolution}

"""
    l2error(sol, exact; component=1)

L2 error of solution component `component` against `exact(x, y)`.
"""
l2error(sol::AnySolution, exact::Function; component::Integer=1) =
    l2error(sol.prob.mesh, sol.u[:, component, :], exact)

Base.show(io::IO, sol::DGSolution) =
    print(io, "DGSolution(", join(size(sol.u), "×"), " at t = ", sol.t, ")")
Base.show(io::IO, sol::HDGSolution) =
    print(io, "HDGSolution(", join(size(sol.u), "×"), ", ", sol.iterations, " GMRES iterations)")
Base.show(io::IO, sol::CGSolution) =
    print(io, "CGSolution(", join(size(sol.u), "×"), ", energy = ", sol.energy, ")")

# -------------------------------------------------------------------- solve

"""
    _ordered_bcs(bc, mesh)

Boundary conditions may be given positionally (a vector/tuple indexed by
boundary tag) or, when the mesh generator attached boundary names, as a
`NamedTuple` keyed by those names, e.g. for `mkmesh_square`:
`(bottom=Dirichlet(), right=Neumann(), top=Dirichlet(), left=Neumann())`.
"""
_ordered_bcs(bc, mesh) = bc
function _ordered_bcs(bc::NamedTuple, mesh)
    names = mesh.boundary_names
    (names === nothing || isempty(names)) &&
        throw(ArgumentError("this mesh carries no boundary names; pass the " *
                            "boundary conditions as a vector ordered by boundary tag"))
    unknown = setdiff(collect(keys(bc)), names)
    isempty(unknown) ||
        throw(ArgumentError("unknown boundary name(s) $(Tuple(unknown)); " *
                            "this mesh's boundaries are $(Tuple(names))"))
    missing_names = setdiff(names, collect(keys(bc)))
    isempty(missing_names) ||
        throw(ArgumentError("missing boundary condition(s) for $(Tuple(missing_names))"))
    return [bc[name] for name in names]
end

# Dirichlet data for HDG/CG: accept a constant or a function of the point
# coordinates ((x, y) in 2D, (x, y, z) in 3D) and produce the
# `dbc(coords::Matrix) -> values` closure the solvers expect.
lower_dbc(bc::Dirichlet) = lower_dbc(bc.value)
lower_dbc(g::Number) = p -> fill(Float64(g), size(p, 1))
lower_dbc(g::Function) = p -> [Float64(g(p[i, :]...)) for i in axes(p, 1)]

_initial_state(prob::DGProblem) = _initial_state(prob.u0, prob)
_initial_state(u0::AbstractArray{<:Any, 3}, prob) = float(copy(u0))
_initial_state(u0::Union{Tuple, AbstractVector}, prob) =
    initu(prob.mesh, nvariables(prob.equation), u0)

"""
    _dg_physics(prob::DGProblem) -> DGPhysics

Assemble the physics bundle the DG kernels consume: the equation, the
per-boundary condition tuple (validated against the mesh's boundary count),
the numerical flux, source, and stabilization — problem-level `nothing`s
resolved to the equation's defaults.
"""
function _dg_physics(prob::DGProblem)
    eq, mesh = prob.equation, prob.mesh
    bcs = _ordered_bcs(prob.bc, mesh)
    tags = @view mesh.f[:, end]     # right element, or -tag on boundary faces
    nbnd = maximum(-tags[tags .< 0])
    length(bcs) == nbnd ||
        throw(ArgumentError("mesh has $nbnd boundaries, got $(length(bcs)) boundary conditions"))
    return DGPhysics(eq;
                     boundary_conditions=Tuple(bcs),
                     numerical_flux=something(prob.numerical_flux,
                                              default_numerical_flux(eq)),
                     source=prob.source,
                     stabilization=prob.stabilization === nothing ?
                                   default_stabilization(eq) : prob.stabilization)
end

# Shared setup for the internal RK4 stepper and the SciML `semidiscretize`
# bridge: build the physics bundle, the (possibly device-resident) DGContext
# and initial state, and select the residual kernel.
function _dg_setup(prob::DGProblem; ArrayT=Array, ngauss=nothing)
    T = eltype(prob)
    phys = _dg_physics(prob)

    master = ngauss === nothing ? Master(prob.mesh) : Master(prob.mesh, ngauss)
    ctx = adapt(ArrayT, DGContext(master, prob.mesh; T))
    u = adapt(ArrayT, T.(_initial_state(prob)))
    phys_d = adapt(ArrayT, phys)

    residual! = has_diffusion(prob.equation) ? rldgexpl! : rinvexpl!
    return ctx, phys_d, u, residual!
end

CommonSolve.solve(prob::DGProblem; kwargs...) = solve(prob, RK4(); kwargs...)

"""
    solve(prob::DGProblem, RK4(); dt, tfinal (or nstep), t0=0.0,
          ArrayT=Array, ngauss=nothing, callback=nothing, restart=nothing)

Run the internal RK4 time loop.

`callback` — any callable `cb(state::SolveState) -> Union{Nothing, Bool}`:
a plain closure, a built-in callback ([`ProgressCallback`](@ref),
[`AnalysisCallback`](@ref), [`SaveSolutionCallback`](@ref),
[`SteadyStateCallback`](@ref), [`CheckpointCallback`](@ref),
[`StepsizeCallback`](@ref)), or a [`CallbackSet`](@ref) composing several.
It is called after every step; `state.u` is the *live* solution array
(device-resident under `ArrayT=CuArray`; copy or `Array(state.u)` it if you
keep it). Return `true` to stop early. Callbacks are observers: with a
fixed `dt` the computed solution is bit-identical with and without them. A
[`StepsizeCallback`](@ref) may write `state.dt`; the loop then advances at
the new step size and, when `tfinal` is given, clamps the last step to land
on `tfinal` exactly. The callback rides back on the solution as
`sol.callbacks`.

`restart` — path to a [`CheckpointCallback`](@ref) file; the solve resumes
from its `u`/`t`/`step` (the problem's `u0` and the `t0` keyword are
ignored, and `nstep` counts the *additional* steps to take).
"""
function CommonSolve.solve(prob::DGProblem, ::RK4;
                           dt, tfinal=nothing, nstep=nothing, t0=0.0,
                           ArrayT=Array, ngauss=nothing, callback=nothing,
                           restart=nothing)
    (tfinal === nothing) == (nstep === nothing) &&
        throw(ArgumentError("give exactly one of `tfinal` or `nstep`"))

    T = eltype(prob)
    ctx, phys, u, residual! = _dg_setup(prob; ArrayT, ngauss)

    step0 = 0
    if restart !== nothing
        chk = load_checkpoint(restart)
        size(chk.u) == size(u) ||
            throw(ArgumentError("checkpoint state is $(size(chk.u)) but the " *
                                "problem needs $(size(u))"))
        copyto!(u, chk.u)
        t0, step0 = chk.t, chk.step
    end
    nsteps = nstep === nothing ? max(round(Int, (tfinal - t0) / dt), 0) : nstep

    if callback === nothing
        rk4_ka!(residual!, ctx, phys, u, T(t0), T(dt), nsteps)
        return DGSolution(Array(u), t0 + nsteps * dt, prob)
    end

    state = SolveState(u, Float64(t0), step0, Float64(dt),
                       tfinal === nothing ? step0 + nsteps : typemax(Int),
                       tfinal === nothing ? NaN : Float64(tfinal),
                       prob, ctx, phys)
    initialize!(callback, state)
    ws = _default_ka_ws(residual!, ctx, phys)
    stages = ntuple(_ -> similar(u), 5)

    # Working-precision time accumulator reproducing the fused rk4_ka! path's
    # `t += dt` arithmetic, so an attached observer changes no bits of u.
    tT = T(t0)
    dt0 = Float64(dt)
    laststep = step0 + nsteps
    dynamic = state.dt != dt0   # a callback owns state.dt: switch to time-based
                                # termination and clamp the last step onto tfinal
    finished() = dynamic && !isnan(state.tfinal) ?
                 state.t ≥ state.tfinal - 1e-12 * max(abs(state.tfinal), 1.0) :
                 state.step ≥ laststep

    while !finished()
        dtstep = state.dt
        dynamic |= dtstep != dt0
        if dynamic && !isnan(state.tfinal)
            dtstep = min(dtstep, state.tfinal - state.t)
        end
        dtstep > 0 ||
            throw(ArgumentError("callback set a non-positive dt = $dtstep"))
        rk4_ka!(residual!, ctx, phys, u, tT, T(dtstep), 1; ws, stages)
        state.step += 1
        state.t += dtstep
        tT += T(dtstep)
        callback(state) === true && break
    end
    finish!(callback, state)
    return DGSolution(Array(u), state.t, prob, callback)
end

"""
    semidiscretize(prob::DGProblem, tspan; ArrayT=Array, ngauss=nothing) -> ODEProblem

Spatial semidiscretization of `prob` as a SciML `ODEProblem`, so any
OrdinaryDiffEq.jl time integrator (adaptive, SSP, IMEX, …) can drive the DG
right-hand side. Requires SciMLBase to be loaded (it usually is, via any
`using OrdinaryDiffEq...` package; otherwise `using SciMLBase`).

```julia
using TwoDG, OrdinaryDiffEqTsit5
ode = semidiscretize(prob, (0.0, 1.0))
sol = solve(ode, Tsit5())
```
"""
function semidiscretize(prob::DGProblem, tspan; kwargs...)
    ext = Base.get_extension(parentmodule(@__MODULE__), :TwoDGSciMLBaseExt)
    ext === nothing &&
        error("semidiscretize requires SciMLBase. Load it first, e.g. " *
              "`using SciMLBase` or any OrdinaryDiffEq solver package.")
    return ext._semidiscretize(prob, tspan; kwargs...)
end

# ------------------------------------------------------------- CFL time step

"""
    compute_dt(prob::DGProblem; cfl=0.3) -> dt

CFL-limited explicit time step: over all elements,

    dt = cfl * min 1 / ( λ (2p+1)/h + κ ((2p+1)/h)² )

with `h` the smallest inscribed-circle (2D) / inscribed-sphere (3D) diameter
(`min_inscribed_diameter`), `p` the polynomial order, `λ` the maximum
characteristic speed of the equation (evaluated from the initial state for
[`EulerEquations`](@ref)), and `κ` its [`diffusivity`](@ref) (LDG diffusion
limits the step quadratically). `cfl = 0.3` is a conservative default for
the internal [`RK4`](@ref) stepper. [`StepsizeCallback`](@ref) applies the
same formula to the *running* solution.
"""
function compute_dt(prob::DGProblem; cfl=0.3)
    mesh = prob.mesh
    pfac = 2 * mesh.porder + 1
    λ = _max_wavespeed(prob.equation, prob)
    κ = diffusivity(prob.equation)
    (λ > 0 || κ > 0) ||
        throw(ArgumentError("equation has neither a propagation speed nor a diffusivity"))
    h = min_inscribed_diameter(mesh)
    return cfl / (λ * pfac / h + κ * (pfac / h)^2)
end

_max_wavespeed(eq::ConvectionEquation, prob) = _max_velocity(eq.velocity, prob.mesh)
_max_wavespeed(eq::ConvectionDiffusionEquation, prob) = _max_velocity(eq.velocity, prob.mesh)
_max_wavespeed(eq::WaveEquation, prob) = abs(eq.c)
_max_wavespeed(::PoissonEquation, prob) = 0.0

function _max_wavespeed(eq::EulerEquations, prob)
    u0 = prob.u0
    u = u0 isa AbstractArray{<:Any, 3} ? u0 : interpolate(prob.mesh, u0)
    NC = Val(nvariables(eq))
    smax = 0.0
    for it in axes(u, 3), i in axes(u, 1)
        state = SVector(ntuple(c -> u[i, c, it], NC))
        smax = max(smax, norm(velocity(eq, state)) + soundspeed(eq, state))
    end
    return smax
end

_max_velocity(v, mesh) = norm(v)
function _max_velocity(v::Function, mesh)
    Dim = Val(ndims(mesh))
    smax = 0.0
    dg = mesh.dgnodes
    for it in axes(dg, 3), i in axes(dg, 1)
        smax = max(smax, norm(v(SVector(ntuple(d -> dg[i, d, it], Dim)))))
    end
    return smax
end

# ------------------------------------------------------------ HDG / CG solves

CommonSolve.solve(prob::HDGProblem; kwargs...) = solve(prob, GMRES(); kwargs...)

function CommonSolve.solve(prob::HDGProblem, ::Direct; ngauss=nothing)
    mesh = prob.mesh
    master = ngauss === nothing ? Master(mesh, 4 * (mesh.porder + 1)) : Master(mesh, ngauss)
    param = hdg_param(prob.equation, prob.stabilization)
    u, q, uhat = hdg_direct_batched(master, mesh, prob.source, lower_dbc(prob.bc), param)
    return HDGSolution(u, q, uhat, 0, prob)
end

function CommonSolve.solve(prob::HDGProblem, alg::GMRES; ngauss=nothing, ArrayT=Array)
    mesh = prob.mesh
    master = ngauss === nothing ? Master(mesh, 4 * (mesh.porder + 1)) : Master(mesh, ngauss)
    param = hdg_param(prob.equation, prob.stabilization)
    driver = alg.batched ? hdg_parsolve_batched : hdg_parsolve
    u, q, uhat, niter = driver(master, mesh, prob.source, lower_dbc(prob.bc), param;
                               ArrayT, restart=alg.restart, tol=alg.tol,
                               maxit=alg.maxit, preconditioner=alg.preconditioner)
    return HDGSolution(u, q, uhat, niter, prob)
end

CommonSolve.solve(prob::CGProblem; kwargs...) = solve(prob, Direct(); kwargs...)

_cg_master(prob, ngauss) =
    ngauss === nothing ? Master(prob.mesh, 4 * prob.mesh.porder) : Master(prob.mesh, ngauss)

function CommonSolve.solve(prob::CGProblem, ::Direct; ngauss=nothing)
    uh, energy = cg_solve(prob.mesh, _cg_master(prob, ngauss), prob.source,
                          cg_param(prob.equation, prob.reaction))
    return CGSolution(reshape(uh, size(uh, 1), 1, size(uh, 2)), energy, 0, prob)
end

function CommonSolve.solve(prob::CGProblem, alg::ConjugateGradient;
                           ngauss=nothing, ArrayT=Array)
    param = cg_param(prob.equation, prob.reaction)
    iszero(param.c) ||
        throw(ArgumentError("ConjugateGradient requires a symmetric operator " *
                            "(no convection); use GMRES()"))
    uh, energy, niter = cg_parsolve(prob.mesh, _cg_master(prob, ngauss),
                                    prob.source, param; ArrayT,
                                    tol=alg.tol, maxit=alg.maxit,
                                    preconditioner=alg.preconditioner)
    return CGSolution(reshape(uh, size(uh, 1), 1, size(uh, 2)), energy, niter, prob)
end

function CommonSolve.solve(prob::CGProblem, alg::GMRES; ngauss=nothing, ArrayT=Array)
    uh, energy, niter = cg_parsolve(prob.mesh, _cg_master(prob, ngauss),
                                    prob.source, cg_param(prob.equation, prob.reaction);
                                    ArrayT, tol=alg.tol, maxit=alg.maxit,
                                    restart=alg.restart,
                                    preconditioner=alg.preconditioner)
    return CGSolution(reshape(uh, size(uh, 1), 1, size(uh, 2)), energy, niter, prob)
end

end # module Interface
