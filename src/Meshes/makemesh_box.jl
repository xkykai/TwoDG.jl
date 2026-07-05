using TwoDG

"""
    make_box_mesh(m=2, n=2, o=2; lengths=(1.0, 1.0, 1.0)) -> (p, t)

Structured tetrahedral mesh of the box `[0,L₁]×[0,L₂]×[0,L₃]` on an
`m × n × o` vertex grid: each grid cell is split into **6 tetrahedra around
its main diagonal (Kuhn/Freudenthal triangulation)** — unlike the 5-tet
split it is conforming without parity alternation and refines uniformly.
Returns vertices `p (np, 3)` and positively oriented tetrahedra `t (nt, 4)`.
"""
function make_box_mesh(m::Int=2, n::Int=2, o::Int=2;
                       lengths::NTuple{3, <:Real}=(1.0, 1.0, 1.0))
    xs = range(0.0, lengths[1], length=m)
    ys = range(0.0, lengths[2], length=n)
    zs = range(0.0, lengths[3], length=o)

    vid(i, j, k) = i + (j - 1) * m + (k - 1) * m * n
    p = zeros(m * n * o, 3)
    for k in 1:o, j in 1:n, i in 1:m
        p[vid(i, j, k), :] .= (xs[i], ys[j], zs[k])
    end

    # Kuhn split: each cell's 6 tets are the monotone lattice paths from the
    # cell corner (0,0,0) to (1,1,1) — one per permutation of the axes. All
    # share the main diagonal, so neighboring cells' face triangulations match.
    perms = ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))
    nt = 6 * (m - 1) * (n - 1) * (o - 1)
    t = zeros(Int, nt, 4)
    e = 0
    for k in 1:(o - 1), j in 1:(n - 1), i in 1:(m - 1)
        for π in perms
            e += 1
            step = [0, 0, 0]
            t[e, 1] = vid(i, j, k)
            for (v, ax) in enumerate(π)
                step[ax] = 1
                t[e, v + 1] = vid(i + step[1], j + step[2], k + step[3])
            end
        end
    end

    # positive orientation (right-handed vertex order): swap two vertices of
    # any tet with negative signed volume
    for e in 1:nt
        a = @view p[t[e, 2], :]
        b = @view p[t[e, 3], :]
        c = @view p[t[e, 4], :]
        v0 = @view p[t[e, 1], :]
        d1 = a - v0
        d2 = b - v0
        d3 = c - v0
        vol6 = d1[1] * (d2[2] * d3[3] - d2[3] * d3[2]) -
               d1[2] * (d2[1] * d3[3] - d2[3] * d3[1]) +
               d1[3] * (d2[1] * d3[2] - d2[2] * d3[1])
        if vol6 < 0
            t[e, 3], t[e, 4] = t[e, 4], t[e, 3]
        end
    end

    return p, t
end

"""
    box_geometry(m=2, n=2, o=2; lengths=(1.0, 1.0, 1.0)) -> MeshGeometry{3}

Geometry of the box `[0,L₁]×[0,L₂]×[0,L₃]` on an `m × n × o` vertex grid
(Kuhn 6-tet cells), with boundaries named `:left`/`:right` (x), `:front`/
`:back` (y), `:bottom`/`:top` (z), tags 1–6.
"""
function box_geometry(m=2, n=2, o=2; lengths=(1.0, 1.0, 1.0))
    p, t = make_box_mesh(m, n, o; lengths=Tuple(lengths))

    ϵ = 1e-6 * maximum(lengths)
    L1, L2, L3 = lengths
    boundaries = (
        left   = p -> p[:, 1] .< ϵ,
        right  = p -> p[:, 1] .> L1 - ϵ,
        front  = p -> p[:, 2] .< ϵ,
        back   = p -> p[:, 2] .> L2 - ϵ,
        bottom = p -> p[:, 3] .< ϵ,
        top    = p -> p[:, 3] .> L3 - ϵ,
    )

    return MeshGeometry(p, t; boundaries)
end

"""
    mkmesh_box(m=2, n=2, o=2, porder=1; lengths=(1.0, 1.0, 1.0)) -> Mesh

Solver-ready tetrahedral mesh of the box:
`discretize(box_geometry(m, n, o; lengths), porder)`.
"""
mkmesh_box(m=2, n=2, o=2, porder=1; lengths=(1.0, 1.0, 1.0)) =
    discretize(box_geometry(m, n, o; lengths), porder)
