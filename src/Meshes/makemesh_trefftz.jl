using TwoDG

"""
    mkmesh_trefftz(m=15, n=30, porder=3, node_spacing_type=0,
                   tparam=[0.1, 0.05, 1.98]) -> Mesh

Curved O-mesh around a Trefftz (Karman–Trefftz) airfoil, built by conformally
mapping an `m × n` structured rectangle through `exp` and the K–T transform.
`tparam = [x0, y0, exponent]` are the airfoil parameters: circle-center
shifts `x0`/`y0` and K–T exponent (`2` gives a Joukowski airfoil). Boundary
tag 1 (`:airfoil`) is the airfoil surface, tag 2 (`:farfield`) the outer
circle. See also [`trefftz`](@ref) for the potential-flow driver.
"""
function mkmesh_trefftz(m=15, n=30, porder=3, node_spacing_type=0, tparam=[0.1, 0.05, 1.98])
    n = 2 * Int(ceil(n / 2))

    p0, t0 = make_square_mesh(m, n ÷ 2, 0)
    p1, t1 = make_square_mesh(m, n ÷ 2, 1)

    nump = size(p0, 1)

    t1 = t1 .+ nump
    p1[:, 2] .+= 1

    p = vcat(p0, p1)
    t = vcat(t0, t1)

    plocal, tlocal = localpnts(porder, node_spacing_type)

    # high-order nodes on the structured rectangle; every subsequent map is
    # conformal and applied to the nodes directly (isoparametric elements)
    dgnodes = straight_dgnodes(p, t, plocal)

    # map the rectangle to an annulus: (x, y) -> exp(2x + iπy)
    p[:, 1] .*= 2
    p[:, 2] .*= π
    w = exp.(p[:, 1] .+ im .* p[:, 2])
    p[:, 1] .= real.(w)
    p[:, 2] .= imag.(w)

    # the two rectangle edges y = 0 and y = 2 land on the same physical points;
    # fixmesh dedupes them (it preserves triangle order, so dgnodes still match)
    p, t = fixmesh(p, t)

    f, t2f, t2o = mkt2f(t)

    fcurved = trues(size(f, 1))
    tcurved = trues(size(t, 1))

    boundary1(p) = vec(sqrt.(sum(p .^ 2, dims=2)) .< 2)
    boundary2(p) = vec(sqrt.(sum(p .^ 2, dims=2)) .> 2)

    f = setbndnbrs(p, f, [boundary1, boundary2])

    # Karman–Trefftz transform of vertices and high-order nodes
    x0, y0, nkt = tparam[1], tparam[2], tparam[3]
    rot = atan(y0, 1 + x0)
    r = sqrt((1 + x0)^2 + y0^2)

    kt(w) = ((1 + ((w - 1) / (w + 1))^nkt) / (1 - ((w - 1) / (w + 1))^nkt)) * nkt

    w = p[:, 1] .+ im .* p[:, 2]
    w = kt.(r * exp(-im * rot) .* w .- x0 .+ im * y0)
    p[:, 1] .= real.(w)
    p[:, 2] .= imag.(w)

    wd = exp.(2 .* dgnodes[:, 1, :] .+ im .* π .* dgnodes[:, 2, :])
    wd = kt.(r * exp(-im * rot) .* wd .- x0 .+ im * y0)
    dgnodes[:, 1, :] .= real.(wd)
    dgnodes[:, 2, :] .= imag.(wd)

    return TwoDG.Mesh(; p, t, f, t2f, t2o, fcurved, tcurved, porder, plocal, tlocal,
                      dgnodes, elcon=mkelcon(t2f, t2o, porder), f2f=mkf2f(f, t2f),
                      boundary_names=[:airfoil, :farfield])
end
