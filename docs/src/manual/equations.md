# Equations and boundary conditions

## Built-in equations

Equations are small immutable structs; the number of conserved components
comes from [`nvariables`](@ref).

| Equation | Components | Problems |
|---|---|---|
| [`ConvectionEquation`](@ref)`(v)` | 1 | `DGProblem` |
| [`ConvectionDiffusionEquation`](@ref)`(v, κ)` | 1 | `DGProblem` (explicit LDG), `HDGProblem`, `CGProblem` |
| [`WaveEquation`](@ref)`(c)` | 3 | `DGProblem` |
| [`EulerEquations`](@ref)`(γ)` | 4 | `DGProblem` |
| [`PoissonEquation`](@ref)`(κ)` | 1 | `HDGProblem`, `CGProblem` |

The convective velocity of `ConvectionEquation`/`ConvectionDiffusionEquation`
may be a constant 2-vector or a position function `x::SVector{2} -> SVector{2}`
(a rotating field, say) — the function form is supported by the explicit DG
solvers only.

## Numerical fluxes

The surface (numerical) flux is a first-class object, decoupled from the
equation — any callable `(eq, uL, uR, n, x, t) -> SVector`:

- [`RoeFlux`](@ref)`()` — Roe's approximate Riemann solver (default for
  Euler and the wave system),
- [`LaxFriedrichs`](@ref)`()` — local Lax–Friedrichs / Rusanov (default for
  scalar convection; works for any equation with [`flux`](@ref) and
  [`max_abs_speed`](@ref)).

Every equation has a [`default_numerical_flux`](@ref); override it per
problem with `DGProblem(eq, mesh; numerical_flux = LaxFriedrichs(), ...)`.
LDG viscous penalties are likewise a policy object,
[`LDGStabilization`](@ref)`(c11, c11int)`, passed as
`stabilization = LDGStabilization(10.0, 0.0)`.

## Boundary conditions

Boundary conditions are typed objects, one per boundary tag — never integer
codes:

- [`Dirichlet`](@ref)`(g)` — prescribed solution value: a constant, an
  `SVector`, or `(x, t) -> value` (for HDG/CG problems, `(x, y) -> g`),
- [`Neumann`](@ref)`()` — homogeneous natural boundary,
- [`SlipWall`](@ref)`()` — impermeable reflection wall (wave, Euler),
- [`FarField`](@ref)`(state)` — free-stream state, one value per component,
- [`IncomingWave`](@ref)`()` — incoming-wave forcing for the wave system.

For a `DGProblem` pass either a vector ordered by boundary tag, or — when the
mesh carries [`boundary_names`](@ref) — a `NamedTuple` keyed by those names
in any order:

```julia
bc = (bottom = Dirichlet(), right = Neumann(),
      top    = Dirichlet(), left  = Neumann())
prob = DGProblem(eq, mesh; bc, u0)
```

The conditions are carried into the kernels as a tuple, one entry per
boundary tag; a face's integer tag only selects *which* condition applies —
the physics is dispatched on the condition's type, statically per tuple
slot, on CPU and GPU alike.

## Defining your own physics

The physics surface is open: an equation, a numerical flux, or a boundary
condition defined in *your own script* works in every solver path, with no
package edit — see [Extending TwoDG](extending.md) for the contract, and the
[Define your own equation](../tutorials/custom_equation.md) tutorial for an
executable end-to-end example with a convergence check.

## Derived quantities

The Euler primitives ([`pressure`](@ref), [`mach`](@ref),
[`soundspeed`](@ref), [`density`](@ref), [`velocity`](@ref),
[`entropy`](@ref), [`energy_kinetic`](@ref), [`energy_internal`](@ref),
[`energy_total`](@ref)) are pointwise functions `(eq, u::SVector) -> Real`,
evaluated over a whole field with [`derived_field`](@ref)`(f, eq, u)` —
e.g. `scaplot(mesh, derived_field(mach, eq, sol.u))` — and integrated
quadrature-exactly with [`integrate`](@ref) (see
[Callbacks and diagnostics](callbacks.md)). Any user closure with the same
signature composes identically.
