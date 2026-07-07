# Extending TwoDG

The physics surface is open: a new **equation**, **numerical flux**, or
**boundary condition** defined in *your own script*, using only exported
methods, runs in every solver path with no package edit. All contract
methods are pointwise, allocation-free, and `SVector`-based, so the same
definition compiles into the CPU-thread and GPU kernels alike.

The [Define your own equation](../tutorials/custom_equation.md) tutorial
executes everything on this page end-to-end, convergence table included;
the "user-defined physics" testset in `test/test_ka.jl` keeps the contract
honest in CI.

!!! warning "Stability"
    The extension contract is public but pre-1.0; changes are recorded in
    `NEWS.md`.

## A new equation

Subtype [`AbstractEquation`](@ref)`{Dim}` — the `{Dim}` parameter is
mandatory (the kernels and problem constructors dispatch on it) — and
implement, at minimum, [`nvariables`](@ref) and [`flux`](@ref):

```julia
using TwoDG, StaticArrays

struct Burgers <: TwoDG.AbstractEquation{2} end

TwoDG.nvariables(::Burgers) = 1
TwoDG.flux(::Burgers, u::SVector{1}, x, t) = (u .* u ./ 2, u .* u ./ 2)
```

`flux` returns one `SVector{nc}` per space direction, so the conservation
law reads `∂u/∂t + Σ_d ∂f_d/∂x_d = source`. Two conventions matter for
GPU/`Float32` correctness (the same rules as inside the package):

- **Never hardcode `Float64`.** Derive the working type from the arguments
  (`u .* u ./ 2` stays generic; `0.5 * u.^2` promotes to `Float64` and
  silently breaks single-precision and GPU runs). Use `zero(T)`/`one(T)`
  or integer-over-integer literals.
- **Stay allocation-free and type-stable** — states are `SVector`s in and
  out; no heap arrays, no data-dependent container types.

Optional methods unlock features:

| method | unlocks |
|---|---|
| [`max_abs_speed`](@ref)`(eq, u, n, x, t)` | [`LaxFriedrichs`](@ref) dissipation |
| [`wavespeed`](@ref)`(eq, u)` | [`compute_dt`](@ref) and [`StepsizeCallback`](@ref) |
| [`varnames`](@ref)`(eq)` | named components in output |
| [`default_numerical_flux`](@ref)`(eq)` | a `solve` default so users can omit `numerical_flux` |
| [`has_diffusion`](@ref) `= true` + the viscous methods below | second-order (LDG) terms |

```julia
TwoDG.max_abs_speed(::Burgers, u::SVector{1}, n, x, t) = abs(u[1] * (n[1] + n[2]))
TwoDG.wavespeed(::Burgers, u::SVector{1}) = abs(u[1]) * sqrt(2 * one(u[1]))
```

With that, the equation drives the standard API:

```julia
prob = DGProblem(Burgers(), mesh; bc, u0, numerical_flux = LaxFriedrichs())
sol  = solve(prob, RK4(); dt = compute_dt(prob), tfinal = 0.1)
```

Volume sources are a *problem* keyword, not an equation method:
`DGProblem(eq, mesh; source = (u, x, t) -> SVector(...), ...)`.

### Diffusive systems

Second-order terms go through the LDG gradient path. Implement
[`has_diffusion`](@ref)` = true`, [`viscous_flux`](@ref)`(eq, u, q, x, t)`
(with `q::SMatrix{Dim, nc}`, row `d` the ``\partial/\partial x_d``
derivatives), and the face methods [`viscous_numerical_flux`](@ref),
[`boundary_viscous_flux`](@ref), and [`boundary_trace`](@ref).
`src/Equations/convection_diffusion.jl` is the compact worked example, and
[`LDGStabilization`](@ref) is the penalty policy the face methods receive.

## A new numerical flux

A numerical flux is **any callable** `(eq, uL, uR, n, x, t) -> SVector{nc}`
— the built-in [`RoeFlux`](@ref) and [`LaxFriedrichs`](@ref) are themselves
just instances of this contract. A central flux, for example:

```julia
struct CentralFlux end
(::CentralFlux)(eq, uL, uR, n, x, t) =
    (TwoDG.normal_flux(eq, uL, n, x, t) + TwoDG.normal_flux(eq, uR, n, x, t)) / 2

prob = DGProblem(eq, mesh; bc, u0, numerical_flux = CentralFlux())
```

[`normal_flux`](@ref) (the physical flux projected on the unit normal) is
provided for every equation that defines `flux`, so most fluxes are a few
lines. Consistency (`F(u, u, n) = f(u)·n`) is your correctness obligation;
free-stream preservation on a curved mesh is a cheap smoke test.

## A new boundary condition

Subtype [`BoundaryCondition`](@ref) and implement **one** of:

- [`boundary_state`](@ref)`(bc, eq, uL, n, x, t) -> SVector` — the ghost
  state; the problem's numerical flux is then evaluated between `uL` and
  the ghost state (how [`FarField`](@ref) and [`SlipWall`](@ref) work), or
- [`boundary_flux`](@ref)`(bc, eq, numerical_flux, uL, n, x, t) -> SVector`
  — the normal boundary flux directly, for conditions that are naturally
  flux-prescribing.

```julia
struct Inflow{T} <: TwoDG.BoundaryCondition
    value :: T
end
TwoDG.boundary_state(bc::Inflow, eq, uL, n, x, t) = SVector(bc.value)

prob = DGProblem(eq, mesh; bc = (left = Inflow(1.0), right = Neumann(),
                                 top = Neumann(), bottom = Neumann()), u0)
```

The honest GPU constraint: boundary conditions are carried into the kernels
as a **tuple, one entry per boundary tag**, and a face's integer tag selects
its slot at compile time — so a BC struct must be `isbits` (numbers,
`SVector`s, `NTuple`s; a captured function is fine if its captures are
isbits). Diffusive problems additionally see [`boundary_trace`](@ref) and
[`boundary_viscous_flux`](@ref) on the LDG path.

## One definition, every solver

The same equation/BC objects construct [`DGProblem`](@ref),
[`HDGProblem`](@ref), and [`CGProblem`](@ref) — TwoDG's "three methods, one
API" promise extends to user physics wherever the method supports the
equation class (explicit DG for hyperbolic systems; HDG/CG for the
diffusion-dominated problems they discretize). And because the contract is
dimension-parametric, a `Dim`-generic equation runs on triangles and
tetrahedra alike.

Finally: a claim of correctness needs a **convergence test**. Refine `h` at
fixed `p` against an exact or manufactured solution and check the `p+1`
rate, as the [custom-equation tutorial](../tutorials/custom_equation.md)
does — a passing smoke test is not proof of order of accuracy.
