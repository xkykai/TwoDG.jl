module Utils

export 
    unique_rows,
    newton_raphson,
    initu

include("unique_rows.jl")
include("rootfinding.jl")
include("initialize_u.jl")
end