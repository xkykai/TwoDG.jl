module Apps

export 
    App,
    mkapp_convection, mkapp_wave, mkapp_euler, eulereval, 
    riemann_to_canonical, canonical_to_riemann

include("app.jl")
include("convection.jl")
include("wave.jl")
include("euler.jl")

end