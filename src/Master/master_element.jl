"""
uniformlocalpnts 2-d mesh generator for the master element.
[plocal,tlocal]=uniformlocalpnts(porder)

   plocal:    node positions (npl,3)
   tlocal:    triangle indices (nt,3)
   porder:    order of the complete polynomial 
              npl = (porder+1)*(porder+2)/2
 """
function uniformlocalpnts(porder)
    plocal = zeros()
    n = porder + 1
    npl = (porder + 1) * (porder + 2) ÷ 2

    plocal = zeros(npl, 3)
    xs = ys = range(0, 1, length=n)

    i_start = 1
    for i in 1:n
        i_end = i_start + n - i
        plocal[i_start:i_end, 2] .= xs[1:n+1-i]
        plocal[i_start:i_end, 3] .= ys[i]
        plocal[i_start:i_end, 1] .= xs[n+1-i:-1:1]
        i_start = i_end + 1
    end

    tlocal = zeros(Int, porder^2, 3)
    i_start_t = 1
    vertex_start = 1
    for i in 1:porder
        i_end_t = i_start_t + porder - i
        tlocal[i_start_t:i_end_t, 1] .= vertex_start:vertex_start + porder - i
        tlocal[i_start_t:i_end_t, 2] .= vertex_start + 1:vertex_start + porder - i + 1
        tlocal[i_start_t:i_end_t, 3] .= vertex_start + porder - i + 2:vertex_start + 2porder - 2i + 2
        
        i_start_t = i_end_t + 1

        if i_start_t < porder^2
            i_end_t = i_start_t + porder - i - 1
            vertex_start += 1
            tlocal[i_start_t:i_end_t, 1] .= vertex_start:vertex_start + porder - i - 1
            tlocal[i_start_t:i_end_t, 2] .= vertex_start + porder - i + 2:vertex_start + 2porder - 2i + 1
            tlocal[i_start_t:i_end_t, 3] .= vertex_start + porder - i + 1:vertex_start + 2porder - 2i
            i_start_t = i_end_t + 1
        end
        
        vertex_start += porder - i + 1
    end

    return plocal, tlocal
end