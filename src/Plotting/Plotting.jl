module Plotting

export
    meshplot, scaplot, meshplot_curved, save_vtk

# The actual implementations live in the TwoDGMakieExt package extension and
# are loaded automatically once a Makie backend is present. Keeping Makie out
# of the hard dependency graph means installing TwoDG does not compile the
# plotting stack.

const _MAKIE_HINT = """
    requires a Makie backend. Load one first, e.g.

        using CairoMakie   # or GLMakie
    """

"""
    meshplot(mesh; nodes=false, annotate="")

Plot a simplicial mesh. `nodes=true` marks the DG nodes; `annotate` may
contain `'p'` (number the vertices) and/or `'t'` (number the elements).
Requires a Makie backend to be loaded (e.g. `using CairoMakie`).
"""
meshplot(args...; kwargs...) = error("meshplot ", _MAKIE_HINT)

"""
    scaplot(mesh, c; limits=nothing, show_mesh=false, figure_size=(800, 800), title="", cmap=:turbo)

Plot contours of a scalar field `c` of shape `(npl, nt)` on `mesh`.
Requires a Makie backend to be loaded (e.g. `using CairoMakie`).
"""
scaplot(args...; kwargs...) = error("scaplot ", _MAKIE_HINT)

"""
    meshplot_curved(mesh; nodes=false, annotate="", pplot=0, figure_size=(800, 800), title="")

Plot a curved (isoparametric) mesh, subdividing each element with order
`pplot` for display. Requires a Makie backend to be loaded
(e.g. `using CairoMakie`).
"""
meshplot_curved(args...; kwargs...) = error("meshplot_curved ", _MAKIE_HINT)

"""
    save_vtk(mesh, u, filename; names=nothing) -> saved file paths
    save_vtk(sol, filename)

Write a solution as high-order Lagrange VTK cells for ParaView (the 3D
visualization path; works for 2D meshes too). `u` is a
`(npl, nc, nt)` (or `(npl, nt)` scalar) field; `names` optionally labels the
components. The solution-object form names components by
`varnames(sol.prob.equation)`. Curved (isoparametric) elements are written
with their true high-order geometry, which ParaView renders natively.

Implemented in the `TwoDGWriteVTKExt` package extension: available once
WriteVTK.jl is loaded (`using TwoDG, WriteVTK`).
"""
save_vtk(args...; kwargs...) =
    error("save_vtk requires WriteVTK.jl. Load it first: `using WriteVTK`.")

end
