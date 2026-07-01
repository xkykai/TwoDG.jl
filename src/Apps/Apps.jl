module Apps

export
    App,
    mkapp_convection, mkapp_wave, mkapp_euler, eulereval,
    mkapp_convection_diffusion,
    riemann_to_canonical, canonical_to_riemann,
    mkapp_convection_pt, mkapp_wave_pt, mkapp_euler_pt,
    mkapp_convection_diffusion_pt

include("app.jl")
include("convection.jl")
include("wave.jl")
include("euler.jl")
include("convection_diffusion.jl")
include("pointwise.jl")

end