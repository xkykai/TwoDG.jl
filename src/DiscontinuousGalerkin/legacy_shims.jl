# Deprecation shims for the legacy `(master, mesh, app, u, time)` residual
# signatures (roadmap A2.2). The legacy whole-face matrix-flux implementations
# are gone; these shims build (and cache) a `DGContext` and route through the
# KernelAbstractions CPU path, so `rk4(rinvexpl, master, mesh, app, u, ...)`
# keeps working for apps in the pointwise convention. Scheduled for deletion
# one minor release after 0.1.0 — call `rinvexpl!`/`rldgexpl!`/`getq!` (or the
# `Interface` layer) directly in new code.

# One cached context (+ per-nc workspaces) per mesh instance; rebuilt if a
# different master is passed for the same mesh.
const _SHIM_CACHE = IdDict{Any, Any}()

function _shim_entry(master, mesh)
    entry = get(_SHIM_CACHE, mesh, nothing)
    if entry === nothing || entry.master !== master
        entry = (; master, ctx = DGContext(master, mesh),
                 rinv = Dict{Int, RinvWorkspace}(),
                 rldg = Dict{Int, RldgWorkspace}())
        _SHIM_CACHE[mesh] = entry
    end
    return entry
end

_rinv_ws(entry, nc::Int) = get!(() -> RinvWorkspace(entry.ctx, nc), entry.rinv, nc)
_rldg_ws(entry, nc::Int) = get!(() -> RldgWorkspace(entry.ctx, nc), entry.rldg, nc)

function _check_pointwise(app, fname)
    if app.arg isa AbstractDict
        throw(ArgumentError(
            "$fname no longer supports legacy matrix-flux apps (Dict params). " *
            "Build the app with the pointwise constructors " *
            "(`mkapp_convection_pt`, `mkapp_wave_pt`, `mkapp_euler_pt`, " *
            "`mkapp_convection_diffusion_pt`) or use the `DGProblem`/`solve` API."))
    end
    return nothing
end

"""
    rinvexpl(master, mesh, app, u, time)

Deprecated legacy signature for the inviscid DG residual; routes through a
cached [`DGContext`](@ref) and [`rinvexpl!`](@ref). Requires an `app` in the
pointwise flux convention. Prefer `rinvexpl!` (or `solve(DGProblem, RK4())`).
"""
function rinvexpl(master, mesh, app, u, time)
    Base.depwarn("`rinvexpl(master, mesh, app, u, time)` is deprecated; use " *
                 "`rinvexpl!(r, ctx, app, u, time)` with a `DGContext`.", :rinvexpl)
    _check_pointwise(app, "rinvexpl")
    entry = _shim_entry(master, mesh)
    return rinvexpl!(similar(u), entry.ctx, app, u, time; ws=_rinv_ws(entry, Int(app.nc)))
end

"""
    rldgexpl(master, mesh, app, u, time)

Deprecated legacy signature for the LDG (viscous) DG residual; routes through
a cached [`DGContext`](@ref) and [`rldgexpl!`](@ref). Requires an `app` in the
pointwise flux convention. Prefer `rldgexpl!` (or `solve(DGProblem, RK4())`).
"""
function rldgexpl(master, mesh, app, u, time)
    Base.depwarn("`rldgexpl(master, mesh, app, u, time)` is deprecated; use " *
                 "`rldgexpl!(r, ctx, app, u, time)` with a `DGContext`.", :rldgexpl)
    _check_pointwise(app, "rldgexpl")
    entry = _shim_entry(master, mesh)
    return rldgexpl!(similar(u), entry.ctx, app, u, time; ws=_rldg_ws(entry, Int(app.nc)))
end

"""
    getq(master, mesh, app, u, time)

Deprecated legacy signature for the LDG gradient `q (npl, 2, nc, nt)`; routes
through a cached [`DGContext`](@ref) and [`getq!`](@ref). Requires an `app` in
the pointwise flux convention. Prefer `getq!`.
"""
function getq(master, mesh, app, u, time)
    Base.depwarn("`getq(master, mesh, app, u, time)` is deprecated; use " *
                 "`getq!(q, ctx, app, u, time)` with a `DGContext`.", :getq)
    _check_pointwise(app, "getq")
    entry = _shim_entry(master, mesh)
    nc = Int(app.nc)
    q = similar(u, size(u, 1), 2, nc, size(u, 3))
    return getq!(q, entry.ctx, app, u, time; ws=_rldg_ws(entry, nc))
end
