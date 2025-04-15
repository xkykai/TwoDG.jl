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

function App(; nc, pg=false, arg=Dict(), bcm=nothing, bcs=nothing, finvi=nothing, finvb=nothing, finvv=nothing, fvisi=nothing, fvisb=nothing, fvisv=nothing, fvisub=nothing, src=nothing)
    return App(nc, pg, arg, bcm, bcs, finvi, finvb, finvv, fvisi, fvisb, fvisv, fvisub, src)
end

function App(app::App; bcm=nothing, bcs=nothing, src=nothing)
    return App(app.nc, app.pg, app.arg, bcm, bcs, app.finvi, app.finvb, app.finvv, app.fvisi, app.fvisb, app.fvisv, app.fvisub, src)
end
