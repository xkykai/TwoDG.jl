#-----------------------------------------------------------------------------
#  Translated from Python mesh visualization functions
#  Original by Per-Olof Persson and Bradley Froehle
#-----------------------------------------------------------------------------

using LinearAlgebra
using CairoMakie
using GeometryBasics
using Colors
using TwoDG.Masters: shape2d, uniformlocalpnts
using Statistics

"""
    SimplexCollection(p, t; linewidth=0.5, edgecolor=:black, facecolor=RGB(0.8, 0.9, 1.0))

Create a collection of triangles from points `p` and triangle indices `t`.
"""
function SimplexCollection(p, t; linewidth=0.5, edgecolor=:black, facecolor=RGB(0.8, 0.9, 1.0))
    # Create triangles from points and indices
    triangles = [Polygon([Point2f(p[t[i,1],1], p[t[i,1],2]), Point2f(p[t[i,2],1], p[t[i,2],2]), Point2f(p[t[i,3],1], p[t[i,3],2]), Point2f(p[t[i,1],1], p[t[i,1],2])]) for i in 1:size(t,1)]
    return triangles, linewidth, edgecolor, facecolor
end

"""
    DGCollection(dgnodes, bou; linewidth=0.5, edgecolor=:black, facecolor=RGB(0.8, 0.9, 1.0))

Create a collection of curved triangles from discontinuous Galerkin nodes.
"""
function DGCollection(dgnodes, bou; linewidth=0.5, edgecolor=:black, facecolor=RGB(0.8, 0.9, 1.0))
    # Create polygons for each element
    polygons = [Polygon([Point2f(dgnodes[bou[j],1,i], dgnodes[bou[j],2,i]) for j in 1:length(bou)]) for i in 1:size(dgnodes,3)]
    return polygons, linewidth, edgecolor, facecolor
end

"""
    draw_curved_mesh!(ax, mesh, pplot; kwargs...)

Draw a curved mesh on the given axis.
"""
function draw_curved_mesh!(ax, mesh, pplot=0; kwargs...)
    if pplot == 0
        pplot = mesh.porder
    end
    
    plocal, _ = uniformlocalpnts(pplot)
    perm = zeros(Int, (pplot+1, 3))
    aux = [1,2,3,1,2]
    for i in 1:3
        ii = findall(x -> x < 1.0e-6, plocal[:,i])
        jj = sortperm(plocal[ii,aux[i+2]])
        perm[:,i] = ii[jj]
    end

    bou = vcat(perm[:,3], perm[2:end,1], perm[2:end,2])

    shapnodes = shape2d(mesh.porder, mesh.plocal, plocal[:,2:end])
    dgnodes = zeros(Float64, (size(plocal,1), 2, size(mesh.t,1)))
    for i in 1:size(mesh.t,1)
        dgnodes[:,1,i] .= vec(mesh.dgnodes[:,1,i]' * shapnodes[:,1,:])
        dgnodes[:,2,i] .= vec(mesh.dgnodes[:,2,i]' * shapnodes[:,1,:])
    end

    polygons, linewidth, edgecolor, facecolor = DGCollection(dgnodes, bou; kwargs...)
    
    # Draw polygons with specified properties
    poly!(ax, polygons, color=facecolor, strokecolor=edgecolor, strokewidth=linewidth)
    
    return nothing
end

"""
    meshplot(mesh; nodes=false, annotate="")

Plot a simplicial mesh.

# Parameters
- `mesh`: Mesh structure
- `nodes`: Draw markers at each node
- `annotate`: "p" to annotate nodes, "t" to annotate simplices
"""
function meshplot(mesh; nodes=false, annotate="")
    println("Entered meshplot")
    
    p = mesh.p
    t = mesh.t

    fig = Figure()
    ax = Axis(fig[1,1], aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    
    triangles, linewidth, edgecolor, facecolor = SimplexCollection(p, t)
    poly!(ax, triangles, color=facecolor, strokecolor=edgecolor, strokewidth=linewidth)
    
    if nodes
        # Correctly flatten the dgnodes array for plotting
        x_coords = vec(mesh.dgnodes[:,1,:])
        y_coords = vec(mesh.dgnodes[:,2,:])
        scatter!(ax, x_coords, y_coords, color=:black, markersize=8)
    end
    
    if 'p' in annotate
        for i in 1:size(p,1)
            text!(ax, "$(i-1)", position=(p[i,:] .+ [0,0]), color=:red, align=(:center, :center))
        end
    end
    
    if 't' in annotate
        for it in 1:size(t,1)
            pmid = mean(p[t[it,:], :], dims=1)[1,:]
            text!(ax, "$(it-1)", position=pmid, align=(:center, :center))
        end
    end
    
    return fig
end

"""
    scaplot(mesh, c; limits=nothing, show_mesh=false)

Plot contours of a scalar field on a mesh.

# Parameters
- `mesh`: Mesh structure
- `c`: Scalar field of dimension (number_of_local_nodes_per_element, number_of_elements)
- `limits`: Optional [cmin, cmax] for thresholding
- `show_mesh`: Boolean to plot mesh over contours
"""
function scaplot(mesh, c; limits=nothing, show_mesh=false, figure_size=(800, 800), title="", cmap=:turbo)
    # cmap = :plasma  # other options: :RdYlBu, :inferno, :viridis, :magma
    
    fig = Figure(size=figure_size)
    ax = Axis(fig[1,1], aspect=DataAspect(), xlabel="x", ylabel="y", title=title)
    
    if isnothing(limits)
        cmin, cmax = extrema(vec(c))
    else
        cmin, cmax = limits
    end
    
    @info "cmin: $cmin"
    @info "cmax: $cmax"

    # For each element in the mesh
    for i in 1:size(mesh.t, 1)
        # Create triangulation for this element
        # Note: Makie doesn't have a direct equivalent to matplotlib's tricontourf
        # We approximate using a mesh and color mapping
        
        # Get the x, y coordinates and scalar values for this element
        x = mesh.dgnodes[:,1,i]
        y = mesh.dgnodes[:,2,i]
        z = c[:,i]
        
        # Create a triangulation using the local triangulation
        vertices = [Point2f(x[j], y[j]) for j in 1:length(x)]
        faces = [GeometryBasics.TriangleFace{Int}(mesh.tlocal[j,1], mesh.tlocal[j,2], mesh.tlocal[j,3]) for j in 1:size(mesh.tlocal, 1)]
        
        # Create a proper mesh
        element_mesh = GeometryBasics.Mesh(vertices, faces)
        
        # Map the values to vertices
        vertex_values = z  # Values at each vertex
        
        # Draw the mesh with color mapping
        mesh!(ax, element_mesh, color=vertex_values, colormap=cmap, colorrange=(cmin, cmax))
    end
    
    if show_mesh
        draw_curved_mesh!(ax, mesh, 0; facecolor=RGBA(0,0,0,0))
    end
    
    # Add colorbar
    # Colorbar(fig[1,2], colormap=cmap, limits=(cmin, cmax), ticks=LinRange(cmin, cmax, 11))
    Colorbar(fig[1,2], colormap=cmap, limits=(cmin, cmax))
    
    return fig
end

"""
    meshplot_curved(mesh; nodes=false, annotate="", pplot=0)

Plot a curved mesh of triangles.

# Parameters
- `mesh`: Mesh structure
- `nodes`: Draw markers at each node
- `annotate`: "p" to annotate nodes, "t" to annotate simplices
- `pplot`: Order of the polynomial used to display mesh
"""
function meshplot_curved(mesh; nodes=false, annotate="", pplot=0, figure_size=(800, 800), title="")
    p = mesh.p
    t = mesh.t
    
    fig = Figure(size=figure_size)
    ax = Axis(fig[1,1], aspect=DataAspect(), xlabel="x", ylabel="y", title=title)
    hidedecorations!(ax)
    hidespines!(ax)
    
    draw_curved_mesh!(ax, mesh, pplot)
    
    if nodes
        # Correctly flatten the dgnodes array for plotting
        x_coords = vec(mesh.dgnodes[:,1,:])
        y_coords = vec(mesh.dgnodes[:,2,:])
        scatter!(ax, x_coords, y_coords, color=:black, markersize=8)
    end
    
    if 'p' in annotate
        for i in 1:size(p,1)
            text!(ax, "$(i-1)", position=(p[i,:] .+ [0,0]), color=:red, align=(:center, :center))
        end
    end
    
    if 't' in annotate
        for it in 1:size(t,1)
            pmid = mean(p[t[it,:], :], dims=1)[1,:]
            text!(ax, "$(it-1)", position=pmid, align=(:center, :center))
        end
    end

    return fig
end