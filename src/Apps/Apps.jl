module Apps

export 
    App,
    mkapp_convection, mkapp_wave, mkapp_euler

include("app.jl")
include("convection.jl")
include("wave.jl")
include("euler.jl")

end