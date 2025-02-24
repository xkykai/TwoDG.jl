module Master

export 
    uniformlocalpnts, shape2d,
    gaussquad1d, gaussquad2d

include("master_element.jl")
include("gauss_quadratures.jl")
end