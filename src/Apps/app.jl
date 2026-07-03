"""
    App(; nc, pg=false, arg=(;), bcm=nothing, bcs=nothing, src=nothing, ...)

Container describing the physics handed to the DG kernels: number of
components `nc`, flux functions, boundary data, and an optional source.

The flux fields hold the *pointwise* flux convention (see the
`mkapp_*_pt` constructors, e.g. [`mkapp_euler_pt`](@ref)): `finvi`/`finvb`
are the interior/boundary inviscid numerical fluxes, `finvv` the inviscid
volume flux, `fvisi`/`fvisb`/`fvisv`/`fvisub` their viscous (LDG)
counterparts, all operating on `SVector`s at a single point. `pg` marks
fluxes that need the quadrature-point coordinate; `arg` is a `NamedTuple` of
physical parameters; `bcm` maps boundary tags to flux branch codes and `bcs`
carries one data row per code; `src(u, x, arg, t)` is the source term.

Users of the high-level API never build an `App` directly — equations like
[`EulerEquations`](@ref TwoDG.Interface.EulerEquations) lower to one inside
`DGProblem`/`solve`.
"""
mutable struct App{N, P, A, BM, BS, FII, FIB, FIV, FVI, FVB, FVV, FVUB, S}
    nc::N
    pg::P
    arg::A
    bcm::BM
    bcs::BS
    finvi::FII
    finvb::FIB
    finvv::FIV
    fvisi::FVI
    fvisb::FVB
    fvisv::FVV
    fvisub::FVUB
    src::S
end

function App(; nc, pg=false, arg=(;), bcm=nothing, bcs=nothing, finvi=nothing, finvb=nothing, finvv=nothing, fvisi=nothing, fvisb=nothing, fvisv=nothing, fvisub=nothing, src=nothing)
    return App(nc, pg, arg, bcm, bcs, finvi, finvb, finvv, fvisi, fvisb, fvisv, fvisub, src)
end

function App(app::App; bcm=nothing, bcs=nothing, src=nothing)
    return App(app.nc, app.pg, app.arg, bcm, bcs, app.finvi, app.finvb, app.finvv, app.fvisi, app.fvisb, app.fvisv, app.fvisub, src)
end
