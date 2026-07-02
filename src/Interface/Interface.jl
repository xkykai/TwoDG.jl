"""
High-level problem/solve API (roadmap A1). Thin, allocation-light wrappers
around the validated low-level solvers:

```julia
using TwoDG

mesh = mkmesh_square(17, 17, 3, 0, 1)
eq   = EulerEquations(γ = 1.4)
prob = DGProblem(eq, mesh; bc = [FarField(uinf), SlipWall(), FarField(uinf), FarField(uinf)],
                 u0 = [ρ0, ρu0, ρv0, ρE0])
sol  = solve(prob, RK4(); dt = 1e-3, tfinal = 1.0)
err  = l2error(sol, exact)      # or scaplot(sol.prob.mesh, sol.u[:, 1, :])
```

Boundary-condition objects lower to the kernels' compiled `bcm::Vector{Int}` /
`bcs::Matrix` representation (an `Int32` code plus a data row stays the right
GPU format; it is just no longer the *user* API).
"""
module Interface

import CommonSolve
using CommonSolve: solve
using StaticArrays
using Adapt: adapt
using ..Apps: mkapp_convection_pt, mkapp_convection_diffusion_pt,
              mkapp_wave_pt, mkapp_euler_pt
using ..Masters: Master
using ..Meshes: Mesh
using ..Utils: initu
using ..DiscontinuousGalerkin: DGContext, rinvexpl!, rldgexpl!, rk4_ka!,
                               RinvWorkspace, RldgWorkspace
using ..HybridizableDiscontinuousGalerkin: hdg_solve, hdg_parsolve,
                                           hdg_parsolve_batched
using ..ContinuousGalerkin: cg_solve
import ..ContinuousGalerkin: l2error

export solve,
    ConvectionEquation, ConvectionDiffusionEquation, WaveEquation,
    EulerEquations, PoissonEquation, nvariables,
    Dirichlet, Neumann, SlipWall, FarField, IncomingWave,
    DGProblem, HDGProblem, CGProblem,
    RK4, Direct, GMRES

# --------------------------------------------------------------- equations

abstract type AbstractEquation end

"""
    ConvectionEquation(velocity)

Linear scalar convection `u_t + ∇·(v u) = s`. `velocity` is a constant
2-vector or a callable `x::SVector{2} -> SVector{2}`.
"""
struct ConvectionEquation{V} <: AbstractEquation
    velocity::V
end

"""
    ConvectionDiffusionEquation(velocity, κ; c11=1.0, c11int=0.0)

Linear convection-diffusion `u_t + ∇·(v u) = ∇·(κ ∇u) + s`, discretized with
LDG viscous fluxes (`c11`/`c11int` are the boundary/interior stabilization
coefficients). Used by both `DGProblem` (explicit LDG) and `HDGProblem`
(steady HDG, where `c11`/`c11int` are ignored).
"""
struct ConvectionDiffusionEquation{V, T} <: AbstractEquation
    velocity::V
    κ::T
    c11::T
    c11int::T
end
ConvectionDiffusionEquation(velocity, κ; c11=1.0, c11int=0.0) =
    ConvectionDiffusionEquation(velocity, promote(κ, c11, c11int)...)

"""
    WaveEquation(c; k=nothing, f=nothing)

First-order wave system (3 components) with speed `c`. `k` (wave vector) and
`f(c, k, x, t)` are only needed when an [`IncomingWave`](@ref) boundary is
used.
"""
struct WaveEquation{T, K, F} <: AbstractEquation
    c::T
    k::K
    f::F
end
WaveEquation(c; k=nothing, f=nothing) = WaveEquation(c, k, f)

"""
    EulerEquations(; γ=1.4)

Compressible Euler equations (4 components), Roe numerical flux.
"""
struct EulerEquations{T} <: AbstractEquation
    γ::T
end
EulerEquations(; γ=1.4) = EulerEquations(γ)

"""
    PoissonEquation(κ=1.0)

Poisson / pure-diffusion equation `-∇·(κ ∇u) = s`, for `HDGProblem` and
`CGProblem`.
"""
struct PoissonEquation{T} <: AbstractEquation
    κ::T
end
PoissonEquation() = PoissonEquation(1.0)

"""
    nvariables(eq)

Number of conserved components of an equation.
"""
nvariables(::ConvectionEquation) = 1
nvariables(::ConvectionDiffusionEquation) = 1
nvariables(::WaveEquation) = 3
nvariables(::EulerEquations) = 4
nvariables(::PoissonEquation) = 1

_velocity(v::Function) = v
_velocity(v) = SVector{2, Float64}(v[1], v[2])

# ------------------------------------------------------- boundary conditions

abstract type BoundaryCondition end

"""
    Dirichlet(value=0.0)

Prescribed solution value on a boundary. For `DGProblem` the value must be a
constant; for `HDGProblem`/`CGProblem` it may be a function `(x, y) -> g`.
"""
struct Dirichlet{G} <: BoundaryCondition
    value::G
end
Dirichlet() = Dirichlet(0.0)

"""
    Neumann(flux=0.0)

Zero-flux (natural) boundary. Only the homogeneous case is currently
supported by the kernels.
"""
struct Neumann{G} <: BoundaryCondition
    flux::G
end
Neumann() = Neumann(0.0)

"Impermeable slip wall (reflection) for the wave and Euler systems."
struct SlipWall <: BoundaryCondition end

"Far-field boundary carrying the free-stream `state` (one value per component)."
struct FarField{S} <: BoundaryCondition
    state::S
end

"Incoming-wave boundary for [`WaveEquation`](@ref) (uses the equation's `k`, `f`)."
struct IncomingWave <: BoundaryCondition end

# Lowering to the kernels' integer-code convention: `bc_code` selects the
# behavior branch inside the pointwise boundary flux, `bc_state` the data row.
bc_code(::ConvectionEquation, ::Union{Dirichlet, FarField}) = 1
bc_code(::ConvectionDiffusionEquation, ::Dirichlet) = 1
bc_code(::ConvectionDiffusionEquation, ::Neumann) = 2
bc_code(::WaveEquation, ::FarField) = 1
bc_code(::WaveEquation, ::SlipWall) = 2
bc_code(::WaveEquation, ::IncomingWave) = 3
bc_code(::EulerEquations, ::FarField) = 1
bc_code(::EulerEquations, ::SlipWall) = 2
bc_code(eq::AbstractEquation, bc::BoundaryCondition) =
    throw(ArgumentError("$(typeof(bc).name.name) boundaries are not supported for $(typeof(eq).name.name)"))

bc_state(eq::AbstractEquation, bc::FarField) = collect(Float64, bc.state)
function bc_state(eq::AbstractEquation, bc::Dirichlet)
    bc.value isa Number ||
        throw(ArgumentError("DG boundary data must be a constant (got $(typeof(bc.value)))"))
    return fill(Float64(bc.value), nvariables(eq))
end
function bc_state(eq::AbstractEquation, bc::Neumann)
    iszero(bc.flux) ||
        throw(ArgumentError("only homogeneous Neumann boundaries are supported"))
    return zeros(nvariables(eq))
end
bc_state(eq::AbstractEquation, ::Union{SlipWall, IncomingWave}) = zeros(nvariables(eq))

"""
    lower_bcs(eq, bcs) -> (bcm, bcs_matrix)

Lower a collection of [`BoundaryCondition`](@ref)s (indexed by the mesh's
boundary tag) to the kernel representation `bcm::Vector{Int}`,
`bcs::Matrix{Float64}`. Boundaries mapping to the same code must carry the
same data (a limitation of the compiled format).
"""
function lower_bcs(eq::AbstractEquation, bcs)
    nc = nvariables(eq)
    bcm = [bc_code(eq, bc) for bc in bcs]
    nrow = maximum(bcm)
    mat = zeros(nrow, nc)
    seen = falses(nrow)
    for (tag, bc) in enumerate(bcs)
        code = bcm[tag]
        state = bc_state(eq, bc)
        if seen[code] && !isapprox(vec(mat[code, :]), state)
            throw(ArgumentError("boundaries with the same type must currently " *
                                "share the same data (boundary $tag conflicts)"))
        end
        mat[code, :] .= state
        seen[code] = true
    end
    return bcm, mat
end

# Dirichlet data for HDG/CG: accept a constant or a function (x, y) -> g and
# produce the `dbc(coords::Matrix) -> values` closure the solvers expect.
lower_dbc(bc::Dirichlet) = lower_dbc(bc.value)
lower_dbc(g::Number) = p -> fill(Float64(g), size(p, 1))
lower_dbc(g::Function) = p -> [Float64(g(p[i, 1], p[i, 2])) for i in axes(p, 1)]

# --------------------------------------------------- equations -> pointwise app

has_diffusion(::AbstractEquation) = false
has_diffusion(::ConvectionDiffusionEquation) = true

pointwise_app(eq::ConvectionEquation; bcm, bcs, source) =
    mkapp_convection_pt(_velocity(eq.velocity); bcm, bcs, src=source)
pointwise_app(eq::ConvectionDiffusionEquation; bcm, bcs, source) =
    mkapp_convection_diffusion_pt(_velocity(eq.velocity); kappa=eq.κ,
                                  c11=eq.c11, c11int=eq.c11int,
                                  bcm, bcs, src=source)
pointwise_app(eq::WaveEquation; bcm, bcs, source) =
    mkapp_wave_pt(; c=eq.c, k=eq.k === nothing ? nothing : SVector{2, Float64}(eq.k...),
                  f=eq.f, bcm, bcs, src=source)
pointwise_app(eq::EulerEquations; bcm, bcs, source) =
    mkapp_euler_pt(; gamma=eq.γ, bcm, bcs, src=source)

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

# ------------------------------------------------------------------ problems

"""
    DGProblem(equation, mesh; bc, u0, source=nothing, T=Float64)

Explicit (L)DG semidiscretization of `equation` on `mesh`. `bc` is a
collection of [`BoundaryCondition`](@ref)s indexed by boundary tag; `u0` the
initial condition — either an `(npl, nc, nt)` array or a vector of `nc`
constants / `(x, y) -> value` functions; `source` an optional pointwise source
`(u, x, param, t) -> SVector`. Solve with [`RK4`](@ref):

    solve(prob, RK4(); dt, tfinal (or nstep), t0=0.0, ArrayT=Array)

`ArrayT=CuArray` (with CUDA.jl loaded) runs the whole time loop on the GPU.
"""
struct DGProblem{E <: AbstractEquation, M, B, U, S, T <: AbstractFloat}
    equation :: E
    mesh     :: M
    bc       :: B
    u0       :: U
    source   :: S
end

function DGProblem(equation::AbstractEquation, mesh; bc, u0, source=nothing,
                   T::Type{<:AbstractFloat}=Float64)
    return DGProblem{typeof(equation), typeof(mesh), typeof(bc), typeof(u0),
                     typeof(source), T}(equation, mesh, bc, u0, source)
end

Base.eltype(::DGProblem{<:Any, <:Any, <:Any, <:Any, <:Any, T}) where {T} = T

"""
    HDGProblem(equation, mesh; bc, source=nothing, stabilization=1.0)

Steady convection-diffusion (or Poisson) problem discretized with HDG. `bc`
is a single [`Dirichlet`](@ref) applied on the whole boundary (constant or
`(x, y) -> g`); `source` is `nothing` or a function of the quadrature-point
coordinate matrix `p -> values`. Solve with [`Direct`](@ref) (sparse LU) or
[`GMRES`](@ref) (default).
"""
struct HDGProblem{E <: AbstractEquation, M, B, S, T}
    equation      :: E
    mesh          :: M
    bc            :: B
    source        :: S
    stabilization :: T
end

HDGProblem(equation::AbstractEquation, mesh; bc, source=nothing, stabilization=1.0) =
    HDGProblem(equation, mesh, bc, source, stabilization)

function hdg_param(eq::ConvectionDiffusionEquation, taud)
    eq.velocity isa Function &&
        throw(ArgumentError("HDGProblem requires a constant convective velocity"))
    return Dict(:kappa => eq.κ, :c => collect(Float64, eq.velocity), :taud => taud)
end
hdg_param(eq::PoissonEquation, taud) =
    Dict(:kappa => eq.κ, :c => [0.0, 0.0], :taud => taud)

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

CGProblem(equation::AbstractEquation, mesh; source, reaction=0.0) =
    CGProblem(equation, mesh, source, reaction)

cg_param(eq::PoissonEquation, s) = (; κ=eq.κ, c=[0.0, 0.0], s)
cg_param(eq::ConvectionDiffusionEquation, s) =
    (; κ=eq.κ, c=collect(Float64, eq.velocity), s)

# ------------------------------------------------------------------ solutions

"""
Solution of a [`DGProblem`](@ref): `u (npl, nc, nt)` at time `t`.
"""
struct DGSolution{U, T, P}
    u    :: U
    t    :: T
    prob :: P
end

"""
Solution of an [`HDGProblem`](@ref): `u (npl, 1, nt)`, flux `q (npl, 2, nt)`,
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
Solution of a [`CGProblem`](@ref): `u (npl, 1, nt)` (DG-numbered for plotting)
and the discrete `energy`.
"""
struct CGSolution{U, E, P}
    u      :: U
    energy :: E
    prob   :: P
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

_initial_state(prob::DGProblem, app) = _initial_state(prob.u0, prob, app)
_initial_state(u0::AbstractArray{<:Any, 3}, prob, app) = float(copy(u0))
_initial_state(u0::Union{Tuple, AbstractVector}, prob, app) =
    initu(prob.mesh, app, u0)

CommonSolve.solve(prob::DGProblem; kwargs...) = solve(prob, RK4(); kwargs...)

function CommonSolve.solve(prob::DGProblem, ::RK4;
                           dt, tfinal=nothing, nstep=nothing, t0=0.0,
                           ArrayT=Array, ngauss=nothing)
    (tfinal === nothing) == (nstep === nothing) &&
        throw(ArgumentError("give exactly one of `tfinal` or `nstep`"))
    if nstep === nothing
        nstep = round(Int, (tfinal - t0) / dt)
    end

    eq, mesh = prob.equation, prob.mesh
    T = eltype(prob)
    bcm, bcs = lower_bcs(eq, prob.bc)
    app = pointwise_app(eq; bcm, bcs, source=prob.source)

    master = ngauss === nothing ? Master(mesh) : Master(mesh, ngauss)
    ctx = adapt(ArrayT, DGContext(master, mesh; T))
    u = adapt(ArrayT, T.(_initial_state(prob, app)))
    app_d = adapt(ArrayT, app)

    residual! = has_diffusion(eq) ? rldgexpl! : rinvexpl!
    rk4_ka!(residual!, ctx, app_d, u, T(t0), T(dt), nstep)

    return DGSolution(Array(u), t0 + nstep * dt, prob)
end

CommonSolve.solve(prob::HDGProblem; kwargs...) = solve(prob, GMRES(); kwargs...)

function CommonSolve.solve(prob::HDGProblem, ::Direct; ngauss=nothing)
    mesh = prob.mesh
    master = ngauss === nothing ? Master(mesh, 4 * (mesh.porder + 1)) : Master(mesh, ngauss)
    param = hdg_param(prob.equation, prob.stabilization)
    u, q, uhat = hdg_solve(master, mesh, prob.source, lower_dbc(prob.bc), param)
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

function CommonSolve.solve(prob::CGProblem, ::Direct; ngauss=nothing)
    mesh = prob.mesh
    master = ngauss === nothing ? Master(mesh, 4 * mesh.porder) : Master(mesh, ngauss)
    uh, energy = cg_solve(mesh, master, prob.source,
                          cg_param(prob.equation, prob.reaction))
    return CGSolution(reshape(uh, size(uh, 1), 1, size(uh, 2)), energy, prob)
end

end # module Interface
