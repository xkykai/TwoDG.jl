# Deprecated constructors for the retired legacy matrix-flux apps (roadmap
# A2.2). The whole-face matrix-flux implementations (Dict params, coordinate-
# matrix fluxes) were deleted; the pointwise `SVector` convention in
# pointwise.jl is the single physics implementation. These stubs keep the old
# names resolvable and point callers at the replacement.

function _legacy_app_error(old, new, example)
    error("`$old` built a legacy matrix-flux app, which has been removed. " *
          "Use `$new` instead (pointwise flux convention — parameters are " *
          "keyword arguments, not `app.arg[:…] = …` mutations), e.g.\n\n    $example\n\n" *
          "or the high-level `DGProblem`/`solve` API.")
end

"""
    mkapp_convection()

Removed. Use [`mkapp_convection_pt`](@ref) (or `ConvectionEquation` +
`DGProblem`).
"""
mkapp_convection() = _legacy_app_error(
    "mkapp_convection", "mkapp_convection_pt",
    "app = mkapp_convection_pt(SVector(1.0, 2.0); bcm, bcs)")

"""
    mkapp_wave()

Removed. Use [`mkapp_wave_pt`](@ref) (or `WaveEquation` + `DGProblem`).
"""
mkapp_wave() = _legacy_app_error(
    "mkapp_wave", "mkapp_wave_pt",
    "app = mkapp_wave_pt(; c=1.0, k=SVector(3.0, 0.0), f=(c, k, x, t) -> …, bcm, bcs)")

"""
    mkapp_euler()

Removed. Use [`mkapp_euler_pt`](@ref) (or `EulerEquations` + `DGProblem`).
"""
mkapp_euler() = _legacy_app_error(
    "mkapp_euler", "mkapp_euler_pt",
    "app = mkapp_euler_pt(; gamma=1.4, bcm, bcs)")

"""
    mkapp_convection_diffusion()

Removed. Use [`mkapp_convection_diffusion_pt`](@ref) (or
`ConvectionDiffusionEquation` + `DGProblem`).
"""
mkapp_convection_diffusion() = _legacy_app_error(
    "mkapp_convection_diffusion", "mkapp_convection_diffusion_pt",
    "app = mkapp_convection_diffusion_pt(SVector(0.0, 0.0); kappa=0.01, c11=10.0, bcm, bcs)")
