# Package-quality checks (B8): method ambiguities, undefined exports, stale
# dependencies, compat bounds, type piracy.

using TwoDG
using Aqua
using Test

@testset "Aqua" begin
    # Statistics is used only by the TwoDGMakieExt extension, which Aqua's
    # stale-deps check does not load.
    Aqua.test_all(TwoDG; ambiguities=(; recursive=false),
                  stale_deps=(; ignore=[:Statistics]))
end
