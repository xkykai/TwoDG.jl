# Package-quality checks (B8): method ambiguities, undefined exports, stale
# dependencies, compat bounds, type piracy.

using TwoDG
using Aqua
using Test

@testset "Aqua" begin
    Aqua.test_all(TwoDG; ambiguities=(; recursive=false))
end
