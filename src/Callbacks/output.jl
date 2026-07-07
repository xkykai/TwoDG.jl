"""
    SaveSolutionCallback(; schedule=IterationInterval(100), interval=nothing,
                         path="out", prefix="solution", fields=(:u,),
                         save_initial=true, save_final=true)

Snapshot writer: at `t0` (if `save_initial`), at each firing, and at the end
of the run (if `save_final`), copies the requested fields to the host and
serializes them to `path/prefix_<step>.jls` (stdlib `Serialization`; read
back with `Serialization.deserialize(file)`, which returns a `NamedTuple`
with `t`, `step`, and one entry per field). Written paths are collected in
`cb.files`.

`fields` entries are either `:u` (the full conserved array) or
`name => f` pairs of pointwise functions with the [`derived_field`](@ref)
contract, e.g. `fields = (:u, :p => pressure, :M => mach)`.
"""
struct SaveSolutionCallback{S <: AbstractSchedule, F <: Tuple}
    schedule     :: S
    path         :: String
    prefix       :: String
    fields       :: F
    save_initial :: Bool
    save_final   :: Bool
    files        :: Vector{String}
end

function SaveSolutionCallback(; schedule::AbstractSchedule = IterationInterval(100),
                              interval = nothing, path::AbstractString = "out",
                              prefix::AbstractString = "solution",
                              fields = (:u,), save_initial::Bool = true,
                              save_final::Bool = true)
    flds = fields isa Tuple ? fields : (fields,)
    all(f -> f === :u || f isa Pair{Symbol, <:Any}, flds) ||
        throw(ArgumentError("fields must be `:u` or `name => pointwise_function` pairs"))
    return SaveSolutionCallback(_schedule(schedule, interval), String(path),
                                String(prefix), flds, save_initial,
                                save_final, String[])
end

function initialize!(cb::SaveSolutionCallback, state)
    initialize!(cb.schedule, state)
    cb.save_initial && _save_snapshot!(cb, state)
    return nothing
end

function (cb::SaveSolutionCallback)(state)
    cb.schedule(state) || return nothing
    _save_snapshot!(cb, state)
    return nothing
end

function finish!(cb::SaveSolutionCallback, state)
    cb.save_final && !_already_saved(cb, state) && _save_snapshot!(cb, state)
    return nothing
end

_already_saved(cb::SaveSolutionCallback, state) =
    !isempty(cb.files) && endswith(cb.files[end], _snapshot_name(cb, state.step))

_snapshot_name(cb::SaveSolutionCallback, step) =
    @sprintf("%s_%06d.jls", cb.prefix, step)

function _save_snapshot!(cb::SaveSolutionCallback, state)
    mkpath(cb.path)
    uh = Array(state.u)
    eq = _equation(state)
    payload = Pair{Symbol, Any}[:t => state.t, :step => state.step]
    for fld in cb.fields
        if fld === :u
            push!(payload, :u => uh)
        else
            name, f = fld
            push!(payload, name => derived_field(f, eq, uh))
        end
    end
    file = joinpath(cb.path, _snapshot_name(cb, state.step))
    serialize(file, (; payload...))
    push!(cb.files, file)
    return nothing
end

"""
    CheckpointCallback(; path="checkpoint.jls", schedule=WallTimeInterval(600), interval=nothing)

Restart-file writer: at each firing (every 10 wall-clock minutes by default)
serializes the full solver state — `u` (host copy), `t`, `step`, `dt` — to
the single file `path`, atomically (written to a temporary file first, then
renamed, so a killed run never leaves a torn checkpoint). Resume with

    solve(prob, RK4(); dt, nstep (or tfinal), restart = path, ...)

which restores `u`/`t`/`step` from the file and continues; a resumed run
reproduces the uninterrupted one to floating-point tolerance.
"""
struct CheckpointCallback{S <: AbstractSchedule}
    schedule :: S
    path     :: String
end

CheckpointCallback(; path::AbstractString = "checkpoint.jls",
                   schedule::AbstractSchedule = WallTimeInterval(600.0),
                   interval = nothing) =
    CheckpointCallback(_schedule(schedule, interval), String(path))

initialize!(cb::CheckpointCallback, state) = initialize!(cb.schedule, state)

function (cb::CheckpointCallback)(state)
    cb.schedule(state) || return nothing
    _write_checkpoint(cb.path, state)
    return nothing
end

function _write_checkpoint(path::String, state)
    dir = dirname(abspath(path))
    isdir(dir) || mkpath(dir)
    tmp = path * ".tmp"
    serialize(tmp, (; u = Array(state.u), t = state.t, step = state.step,
                    dt = state.dt))
    mv(tmp, path; force = true)
    return nothing
end

# read side of CheckpointCallback, consumed by `solve(...; restart = path)`
function load_checkpoint(path::AbstractString)
    chk = deserialize(String(path))
    chk isa NamedTuple && haskey(chk, :u) && haskey(chk, :t) && haskey(chk, :step) ||
        throw(ArgumentError("$path is not a TwoDG checkpoint file"))
    return chk
end
