module Utils

export
    unique_rows,
    newton_raphson,
    initu, interpolate

include("unique_rows.jl")
include("rootfinding.jl")
include("initialize_u.jl")
end