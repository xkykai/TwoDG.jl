# Build the documentation:
#
#     julia --project=docs docs/make.jl
#
# Live-reload preview while writing:
#
#     julia --project=docs -e 'using LiveServer; servedocs()'

using Documenter
using Literate
using TwoDG

# The package will be renamed before registration (ROADMAP B1). Name/URL
# strings are centralized here; the gallery URLs in src/index.md are the only
# other place the repository is spelled out.
const PKG_NAME = "TwoDG"
const REPO_SLUG = "xkykai/TwoDG.jl"
const REPO_URL = "https://github.com/$REPO_SLUG"
const CANONICAL_URL = "https://xkykai.github.io/TwoDG.jl"

# Tutorials are Literate.jl scripts; their code runs on every docs build
# (as Documenter @example blocks), so they cannot rot.
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const TUTORIALS_OUT = joinpath(@__DIR__, "src", "tutorials")
for f in filter(endswith(".jl"), readdir(LITERATE_DIR; join=true))
    Literate.markdown(f, TUTORIALS_OUT; flavor=Literate.DocumenterFlavor())
end

# Developer notes are maintained at the repository root; copy them into the
# source tree at build time so they render as pages without being duplicated.
for (file, target) in [("ROADMAP.md", "devdocs/roadmap.md"),
                       ("GPU_PLAN.md", "devdocs/gpu_plan.md"),
                       ("DOCS_PLAN.md", "devdocs/docs_plan.md"),
                       ("REFACTOR_PLAN.md", "devdocs/refactor_plan.md"),
                       ("CALLBACKS_PLAN.md", "devdocs/callbacks_plan.md"),
                       ("THREED_PLAN.md", "devdocs/threed_plan.md"),
                       (joinpath("test", "CONVERGENCE.md"), "devdocs/convergence.md")]
    src = joinpath(@__DIR__, "..", file)
    dst = joinpath(@__DIR__, "src", target)
    mkpath(dirname(dst))
    isfile(src) || continue
    content = read(src, String)
    # The plans link to repo files (src/..., test/...) that do not exist
    # inside the docs site; unlink them (keep the text) so the strict
    # local-link check passes. Web links and same-page anchors survive.
    content = replace(content, r"\[([^\]\[]+)\]\((?!https?://|#)[^)]+\)" => s"\1")
    write(dst, content)
end

makedocs(;
    modules = [TwoDG],
    sitename = "$PKG_NAME.jl",
    authors = "Xin Kai Lee and contributors",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = CANONICAL_URL,
        edit_link = "main",
        size_threshold = 400 * 2^10,        # devdocs pages are long
        size_threshold_warn = 200 * 2^10,
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Tutorials" => [
            "HDG Poisson and superconvergence" => "tutorials/hdg_poisson.md",
            "Convection–diffusion with LDG" => "tutorials/convection_diffusion_ldg.md",
            "Define your own equation" => "tutorials/custom_equation.md",
            "Callbacks and diagnostics" => "tutorials/callbacks.md",
            "3D transport on a tetrahedral box" => "tutorials/threed_convection.md",
            "3D Euler and the GPU" => "tutorials/threed_euler_gpu.md",
            "HDG superconvergence in 3D" => "tutorials/threed_hdg.md",
        ],
        "Manual" => [
            "Meshes" => "manual/meshes.md",
            "Equations and boundary conditions" => "manual/equations.md",
            "Extending TwoDG" => "manual/extending.md",
            "Solvers" => "manual/solvers.md",
            "Callbacks and diagnostics" => "manual/callbacks.md",
            "3D in TwoDG" => "manual/threed.md",
            "GPU support" => "manual/gpu.md",
            "Plotting" => "manual/plotting.md",
        ],
        "Reference" => [
            "Public API" => "reference/api.md",
            "Internals" => "reference/internals.md",
        ],
        "Developer notes" => [
            "Roadmap" => "devdocs/roadmap.md",
            "Convergence contract" => "devdocs/convergence.md",
            "GPU plan" => "devdocs/gpu_plan.md",
            "Refactor plan" => "devdocs/refactor_plan.md",
            "Callbacks plan" => "devdocs/callbacks_plan.md",
            "3D plan" => "devdocs/threed_plan.md",
            "Documentation plan" => "devdocs/docs_plan.md",
        ],
    ],
    checkdocs = :exports,
    doctest = true,
)

deploydocs(;
    repo = "github.com/$REPO_SLUG.git",
    devbranch = "main",
    push_preview = true,
)
