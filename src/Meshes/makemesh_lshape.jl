using TwoDG

"""
mkmesh_lshape creates 2d mesh data structure for unit l-shape.
mesh=mkmesh_lshape(m,porder,parity)

   mesh:      mesh structure
   m:         number of points for each of the three subsquares
   porder:    polynomial order of approximation (default=1)
   parity:    flag determining the the triangular pattern
              flag = 0 (diagonals sw - ne) (default)
              flag = 1 (diagonals nw - se)

see also: squaremesh, mkt2f, setbndnbrs, uniformlocalpnts, createnodes
"""
function mkmesh_lshape(m=2, porder=1, parity=0, nodetype=1)
    @assert m >= 2
    
    p, t = make_square_mesh(m, m, parity)
    num_p = size(p, 1)
    p = p .* 0.5
    
    p = vcat(
        p,
        hcat(p[:,1], p[:,2] .+ 0.5),
        hcat(p[:,1] .+ 0.5, p[:,2] .+ 0.5)
    )
    
    t = vcat(t, t .+ num_p, t .+ 2 * num_p)
    
    p, t = fixmesh(p, t)
    f, t2f = mkt2f(t)
    
    bndexpr = [p -> p[:,2] .< 1.0e6]
    f = setbndnbrs(p, f, bndexpr)
    
    fcurved = fill(false, size(f, 1))
    tcurved = fill(false, size(t, 1))
    
    plocal, tlocal = localpnts(porder, nodetype)
    
    mesh = TwoDG.Mesh(; p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal)
    mesh = createnodes(mesh)
    mesh = cgmesh(mesh)
    
    return mesh
end