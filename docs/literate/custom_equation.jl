# # Define your own equation
#
# TwoDG's physics surface is open: a new equation — and, along the way, a
# new boundary condition — defined in *this script*, using only exported
# methods, runs through the standard [`DGProblem`](@ref) + [`solve`](@ref)
# path on CPU and GPU alike. This tutorial implements the 2D **Burgers
# equation**
#
# ```math
# \partial_t u + \partial_x \tfrac{u^2}{2} + \partial_y \tfrac{u^2}{2} = 0,
# ```
#
# verifies it against an exact characteristics solution, and — because a
# smoke test is not proof of accuracy — measures its convergence rate.
# The contract itself is documented in
# [Extending TwoDG](../manual/extending.md).

using TwoDG
using StaticArrays
using Printf

# ## The equation
#
# Subtype [`AbstractEquation`](@ref)`{2}` (the dimension is a type
# parameter) and implement the contract. Note the genericity rule: no
# `Float64` literals — `u .* u ./ 2` keeps the element type of `u`, so the
# same definition runs in `Float32` and on GPUs.

struct Burgers <: TwoDG.AbstractEquation{2} end

TwoDG.nvariables(::Burgers) = 1
TwoDG.varnames(::Burgers) = (:u,)
TwoDG.flux(::Burgers, u::SVector{1}, x, t) = (u .* u ./ 2, u .* u ./ 2)

# The flux Jacobian is `(u, u)`, so the characteristic speed along a unit
# normal `n` is `|u (n₁ + n₂)|` — providing it unlocks
# [`LaxFriedrichs`](@ref) dissipation — and the direction-independent bound
# `√2 |u|` unlocks [`compute_dt`](@ref):

TwoDG.max_abs_speed(::Burgers, u::SVector{1}, n, x, t) = abs(u[1] * (n[1] + n[2]))
TwoDG.wavespeed(::Burgers, u::SVector{1}) = sqrt(2 * one(u[1])) * abs(u[1])

# ## An exact solution
#
# For initial data that depends only on ``s = x + y``, Burgers reduces to
# ``\partial_t u + 2 u\, \partial_s u = 0``, whose characteristics give the
# implicit solution ``u = g(s - 2 u t)``. With a Gaussian profile ``g`` and
# ``t`` well before the shock time, the fixed-point iteration converges
# geometrically:

g(s) = 0.5 * exp(-30 * (s - 0.8)^2)

function exact(x, y, t)
    s = x + y
    u = g(s)
    for _ in 1:100
        u = g(s - 2 * u * t)
    end
    return u
end;

# ## A boundary condition, too
#
# The profile is a ridge along the line ``x + y = 0.8``, so it crosses the
# domain boundary — the boundary needs correct inflow data, or the boundary
# error caps the convergence rate. A user boundary condition is a
# [`BoundaryCondition`](@ref) subtype whose [`boundary_state`](@ref) returns
# the ghost state the numerical flux sees; here we prescribe the exact
# solution (which is also exactly how manufactured-solution verification of
# boundary handling works):

struct ExactState <: TwoDG.BoundaryCondition end
TwoDG.boundary_state(::ExactState, eq, uL, n, x, t) = SVector(exact(x[1], x[2], t))

# ## Solve
#
# The user equation drives the standard API — mesh, problem, `solve` — like
# any built-in one. `mkmesh_square` names its four boundaries, and the same
# condition applies to each:

tfinal = 0.05
bc = (bottom = ExactState(), right = ExactState(),
      top = ExactState(), left = ExactState())

mesh = mkmesh_square(17, 17, 2, 0, 1)
prob = DGProblem(Burgers(), mesh; bc, u0 = [(x, y) -> g(x + y)],
                 numerical_flux = LaxFriedrichs())
compute_dt(prob)   # the CFL bound from the wavespeed we defined

#-

sol = solve(prob, RK4(); dt = 5e-4, tfinal)
l2error(sol, (x, y) -> exact(x, y, tfinal))

# ## Convergence
#
# Refine `h` at fixed `p` and check the design rate `p + 1`. The fixed
# `dt = 5·10⁻⁴` sits below the CFL limit on every grid here, and the O(dt⁴)
# time error is far below the spatial one, so the measured rates are purely
# spatial.

function burgers_error(m, porder)
    mesh = mkmesh_square(m, m, porder, 0, 1)
    prob = DGProblem(Burgers(), mesh; bc, u0 = [(x, y) -> g(x + y)],
                     numerical_flux = LaxFriedrichs())
    sol = solve(prob, RK4(); dt = 5e-4, tfinal)
    return l2error(sol, (x, y) -> exact(x, y, tfinal))
end

ms = [9, 17, 33]
for porder in (1, 2)
    errs = [burgers_error(m, porder) for m in ms]
    @printf "p = %d      h        ‖u - uₕ‖      rate\n" porder
    for i in eachindex(ms)
        h = 1 / (ms[i] - 1)
        if i == 1
            @printf "      %8.4f   %11.3e      --\n" h errs[i]
        else
            r = log(errs[i-1] / errs[i]) / log((ms[i] - 1) / (ms[i-1] - 1))
            @printf "      %8.4f   %11.3e   %5.2f\n" h errs[i] r
        end
    end
    println()
end

# Both orders approach their design rate `p + 1` — the acid test that the
# extension surface really is the same surface the built-in equations use.
#
# ## Notes
#
# - A custom **numerical flux** is any callable
#   `(eq, uL, uR, n, x, t) -> SVector`; pass it as the problem's
#   `numerical_flux`. See [Extending TwoDG](../manual/extending.md).
# - On GPUs the same definitions work unchanged
#   (`solve(...; ArrayT = CuArray)`) provided the equation/BC structs are
#   isbits and the methods stay allocation-free — both true here.
# - Past the shock time (`t* ≈ 0.2` for this profile) the solution is no
#   longer smooth and a plain high-order method oscillates; limiting/shock
#   capturing is outside TwoDG's current scope.
