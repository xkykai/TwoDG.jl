using TwoDG
using CSV
using DataFrames
using CairoMakie
using LinearAlgebra

function mkmesh_circle(siz=0.4, porder=3, nodetype=0)
    p, t = make_circle_mesh(siz)
    p, t = fixmesh(p, t)
    
    f, t2f = mkt2f(t)
    
    boundary(p) = sqrt.(sum(p.^2, dims=2)) .> 1 - 2e-2
    bndexpr = [boundary]
    
    f = setbndnbrs(p, f, bndexpr)
    
    fcurved = f[:, 4] .< 0
    tcurved = falses(size(t, 1))
    tcurved[f[fcurved, 3]] .= true
    
    plocal, tlocal = localpnts(porder, nodetype)
    
    mesh = TwoDG.Mesh(; p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal)

    fd_circle(p) = sqrt(sum(p.^2)) - 1
    fds = [fd_circle]
    
    mesh = createnodes(mesh, fds)
    mesh = cgmesh(mesh)

    return mesh
end