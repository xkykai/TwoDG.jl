using TwoDG
using CSV
using DataFrames
using CairoMakie
using LinearAlgebra

function mkmesh_square(m=2, n=2, porder=1, parity=0, nodetype=0)
    p, t = make_square_mesh(m, n, parity)
    p, t = fixmesh(p, t)
    
    f, t2f = mkt2f(t)
    
    boundary_ϵ = 2e-2
    boundary_left(p) = (p[:, 1]) .< boundary_ϵ
    boundary_right(p) = (p[:, 1]) .> 1 - boundary_ϵ
    boundary_bottom(p) = (p[:, 2]) .< boundary_ϵ
    boundary_top(p) = (p[:, 2]) .> 1 - boundary_ϵ
    
    bndexpr = [boundary_left, boundary_right, boundary_bottom, boundary_top]
    
    f = setbndnbrs(p, f, bndexpr)
    
    fcurved = falses(size(f, 1))
    tcurved = falses(size(t, 1))
    tcurved[f[fcurved, 3]] .= true
    
    plocal, tlocal = localpnts(porder, nodetype)
    
    mesh = TwoDG.Mesh(; p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal)
    
    mesh = createnodes(mesh)
    mesh = cgmesh(mesh)

    return mesh
end