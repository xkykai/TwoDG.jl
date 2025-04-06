module Apps

export 
    App,
    mkapp_convection, mkapp_wave

include("app.jl")
include("convection.jl")
include("wave.jl")

end