using LinearAlgebra
using SparseArrays
using TwoDG.Masters: koornwinder1d, koornwinder2d

# HDG method for the 2D incompressible Navier-Stokes equations following
# Nguyen, Peraire & Cockburn, "An implicit high-order hybridizable discontinuous
# Galerkin method for the incompressible Navier-Stokes equations", JCP 230 (2011).
#
# The velocity gradient - pressure - velocity formulation is discretized:
#     L - ∇u = 0
#     ∂u/∂t - ν∇·L + ∇p + ∇·(u⊗u) = f
#     ∇·u = 0
# with the single-valued numerical trace û = λ on faces and total-stress flux
#     f̂ = (-νL + pI + λ⊗λ)·n + τ(u - λ).
# Following solution strategy A of the paper (§4.2), the velocity gradient is
# eliminated element-by-element through the gradient equation
#     L_ij = M⁻¹ (E_j λ_i - C_jᵀ u_i),
# the remaining (u, p) local system is statically condensed, and the globally
# coupled unknowns are the trace of the velocity λ and the mean of the pressure
# ρ_K on each element, giving the saddle-point system
#     [A Bᵀ; B 0] [λ; ρ] = [R; 0]
# closed with the global zero-mean pressure gauge Σ_K |K| ρ_K = 0.
#
# Conventions: Lij = ∂u_i/∂x_j.  Element-local trace DOFs live in the
# face-scalar space of the 3*nps face nodes (edge s, face node a -> index
# (s-1)*nps + a), one block per velocity component; globally, face f, face node
# k, and component c give the interleaved index 2*((f-1)*nps + k - 1) + c.

# ------------------------------------------------------------------------------
# Shared element machinery
# ------------------------------------------------------------------------------

"""
    hdg_elem_volume(dg, master)

Volume metric terms of one (possibly curved) element with nodes `dg` (npl×2):
shape values `shap` (npl×nq), the quadrature- and Jacobian-weighted physical
derivative matrices `shapx`, `shapy` (`shapx[m,g] = w_g jac_g ∂φ_m/∂x`), the
mass matrix `M`, the convection-type matrices `Cx[m,n] = (φ_m, ∂x φ_n)` and
`Cy`, and the weighted Jacobian `wjac`.
"""
function hdg_elem_volume(dg, master)
    shap = @view master.shap[:, 1, :]
    shapxi = @view master.shap[:, 2, :]
    shapet = @view master.shap[:, 3, :]
    gw = master.gwgh

    xxi = shapxi' * dg[:, 1]
    xet = shapet' * dg[:, 1]
    yxi = shapxi' * dg[:, 2]
    yet = shapet' * dg[:, 2]
    jac = xxi .* yet .- xet .* yxi
    shapx = shapxi * Diagonal(gw .* yet) .- shapet * Diagonal(gw .* yxi)
    shapy = -shapxi * Diagonal(gw .* xet) .+ shapet * Diagonal(gw .* xxi)
    wjac = gw .* jac

    M = shap * Diagonal(wjac) * shap'
    Cx = shap * shapx'
    Cy = shap * shapy'

    return (; shap, shapx, shapy, M, Cx, Cy, wjac)
end

"""
    hdg_elem_edge(dg, master, s)

Edge metric terms of local edge `s` of the element with nodes `dg`: the volume
node indices `ps` on the edge, the outward unit normal `n1`, `n2`, the
arc-length factor `ds`, and the weighted measure `wds` at the 1D quadrature
points.
"""
function hdg_elem_edge(dg, master, s)
    ps = master.perm[:, s, 1]
    sh1dx = @view master.sh1d[:, 2, :]
    xξ = sh1dx' * dg[ps, 1]
    yξ = sh1dx' * dg[ps, 2]
    ds = sqrt.(xξ .^ 2 .+ yξ .^ 2)
    n1 = yξ ./ ds
    n2 = -xξ ./ ds
    wds = master.gw1d .* ds
    return (; ps, n1, n2, ds, wds)
end

"Face matrix `⟨w μ_a, μ_b⟩_F` in the nps face nodes for a quadrature weight `w`."
facemat(sh1d, w) = sh1d * Diagonal(w) * sh1d'

"""
    hdg_edge_ops(dg, master)

The lifting operators `E1`, `E2` (npl × 3nps) of the gradient equation,
`Ej[m, ℓ] = ⟨μ_ℓ, φ_m n_j⟩_∂K`, mapping face-scalar trace values to volume test
functions.
"""
function hdg_edge_ops(dg, master)
    nps = master.porder + 1
    npl = size(dg, 1)
    sh1d = @view master.sh1d[:, 1, :]
    E1 = zeros(npl, 3nps)
    E2 = zeros(npl, 3nps)
    for s in 1:3
        ed = hdg_elem_edge(dg, master, s)
        cols = (s - 1) * nps .+ (1:nps)
        E1[ed.ps, cols] .+= facemat(sh1d, ed.wds .* ed.n1)
        E2[ed.ps, cols] .+= facemat(sh1d, ed.wds .* ed.n2)
    end
    return E1, E2
end

"Element-local face-scalar values (3nps) of a face field `Θ` (one DOF per face node)."
function gather_face_scalar(Θ, elcon, it, nps)
    v = zeros(3nps)
    for s in 1:3, a in 1:nps
        v[(s - 1) * nps + a] = Θ[elcon[a, s, it]]
    end
    return v
end

"Element-local face-scalar components (3nps each) of the interleaved trace vector `Λ`."
function gather_face_vector(Λ, elcon, it, nps)
    v1 = zeros(3nps)
    v2 = zeros(3nps)
    for s in 1:3, a in 1:nps
        g = elcon[a, s, it]
        v1[(s - 1) * nps + a] = Λ[2g - 1]
        v2[(s - 1) * nps + a] = Λ[2g]
    end
    return v1, v2
end

"""
    boundary_face_quad(mesh, master, i)

Quadrature data of boundary face `i` in global face-node order: the physical
quadrature points `Xq` (nq×2), the weighted measure `wds`, and the face mass
matrix `Tm`.  Used to impose boundary data through its L2 projection P_∂g onto
P_k(F) — nodal interpolation of the data would introduce an O(h^{k+1})
perturbation that destroys the superconvergence of the method.
"""
function boundary_face_quad(mesh, master, i)
    it = mesh.f[i, 3]
    lf = findfirst(x -> abs(x) == i, mesh.t2f[it, :])
    ori = mesh.t2f[it, lf] > 0 ? 1 : 2
    pp = master.perm[:, lf, ori]
    sh1d = @view master.sh1d[:, 1, :]
    sh1dx = @view master.sh1d[:, 2, :]
    coords = mesh.dgnodes[pp, :, it]
    xξ = sh1dx' * coords[:, 1]
    yξ = sh1dx' * coords[:, 2]
    wds = master.gw1d .* sqrt.(xξ .^ 2 .+ yξ .^ 2)
    Xq = sh1d' * coords
    Tm = sh1d * Diagonal(wds) * sh1d'
    return Xq, wds, Tm
end

# ------------------------------------------------------------------------------
# Incompressible Navier-Stokes
# ------------------------------------------------------------------------------

"""
    hdg_ns_elemmat(dg, master, ν, τ, um, λ1e, λ2e, fe, ffun, uolde, dtinv)

Element matrices for one Newton step of the HDG incompressible Navier-Stokes
discretization, linearized about the velocity `um` (npl×2) and the trace
components `λ1e`, `λ2e` (3nps each).  The body force is either the DG field
`fe` (npl×2, integrated exactly) or the function `ffun` (evaluated at the
quadrature points — interpolating it at the nodes instead would spoil the
superconvergence of the method); `uolde` is the velocity at the previous time
level for backward Euler (or nothing), and `dtinv = 1/Δt` (0 for steady state).

The velocity gradient is eliminated analytically and the local (u, p) system is
statically condensed.  Returns `(Ke, Ge, re, crow, Z, area)`: the condensed
trace matrix (6nps×6nps, trace ordered [λ1; λ2]), the mean-pressure coupling
column, the condensed right-hand side, the element-compatibility row
⟨λ·n, 1⟩_∂K, the local solution operator `Z = A⁻¹[B  bρ  r]` for the recovery
of (u, p), and the element area.
"""
function hdg_ns_elemmat(dg, master, ν, τ, um, λ1e, λ2e, fe, ffun, uolde, dtinv)
    nps = master.porder + 1
    npl = size(dg, 1)
    nfc = 3nps
    sh1d = @view master.sh1d[:, 1, :]

    vol = hdg_elem_volume(dg, master)
    (; shap, shapx, shapy, M, Cx, Cy, wjac) = vol
    area = sum(wjac)

    # Newton linearization of the volume convection about um
    u1g = shap' * um[:, 1]
    u2g = shap' * um[:, 2]
    X1 = shapx * Diagonal(u1g) * shap'
    X2 = shapx * Diagonal(u2g) * shap'
    Y1 = shapy * Diagonal(u1g) * shap'
    Y2 = shapy * Diagonal(u2g) * shap'
    Ku = X1 .+ Y2

    # Edge operators in the face-scalar space
    E1 = zeros(npl, nfc)    # ⟨λ, φ n_j⟩_∂K (gradient lifting / continuity)
    E2 = zeros(npl, nfc)
    FN1 = zeros(npl, npl)   # ⟨(volume field) n_j, φ⟩_∂K
    FN2 = zeros(npl, npl)
    Fτ = zeros(npl, npl)    # τ ⟨u, φ⟩_∂K
    Eτ = zeros(npl, nfc)    # τ ⟨λ, φ⟩_∂K
    Bc = [zeros(npl, nfc) for _ in 1:2, _ in 1:2]   # linearized ⟨(λ·n) λ_i, φ⟩
    rcv = [zeros(npl) for _ in 1:2]                 # its Newton constant term
    Hλc = [zeros(nfc, nfc) for _ in 1:2, _ in 1:2]  # the same, tested on faces
    rcf = [zeros(nfc) for _ in 1:2]
    HN1 = zeros(nfc, npl)   # ⟨(volume field) n_j, μ⟩_∂K
    HN2 = zeros(nfc, npl)
    Hτ = zeros(nfc, npl)    # τ ⟨u, μ⟩_∂K
    Hλτ = zeros(nfc, nfc)   # τ ⟨λ, μ⟩_∂K
    crow = zeros(2nfc)      # ⟨λ·n, 1⟩_∂K

    for s in 1:3
        ed = hdg_elem_edge(dg, master, s)
        (; ps, n1, n2, wds) = ed
        cols = (s - 1) * nps .+ (1:nps)

        λg = (sh1d' * λ1e[cols], sh1d' * λ2e[cols])
        λn = λg[1] .* n1 .+ λg[2] .* n2
        ng = (n1, n2)

        T0 = facemat(sh1d, wds)
        Tn1 = facemat(sh1d, wds .* n1)
        Tn2 = facemat(sh1d, wds .* n2)
        Tλn = facemat(sh1d, wds .* λn)

        E1[ps, cols] .+= Tn1
        E2[ps, cols] .+= Tn2
        FN1[ps, ps] .+= Tn1
        FN2[ps, ps] .+= Tn2
        Fτ[ps, ps] .+= τ .* T0
        Eτ[ps, cols] .+= τ .* T0
        HN1[cols, ps] .+= Tn1
        HN2[cols, ps] .+= Tn2
        Hτ[cols, ps] .+= τ .* T0
        Hλτ[cols, cols] .+= τ .* T0

        # Newton linearization of the trace convection (λ·n)λ_i:
        # (λᵐ·n) δλ_i + (δλ·n) λ_iᵐ, with constant term (λᵐ·n) λ_iᵐ
        for i in 1:2, j in 1:2
            Tij = facemat(sh1d, wds .* ng[j] .* λg[i])
            Bc[i, j][ps, cols] .+= Tij
            Hλc[i, j][cols, cols] .+= Tij
            if i == j
                Bc[i, j][ps, cols] .+= Tλn
                Hλc[i, j][cols, cols] .+= Tλn
            end
        end
        for i in 1:2
            rc = sh1d * (wds .* λn .* λg[i])
            rcv[i][ps] .+= rc
            rcf[i][cols] .+= rc
        end

        crow[cols] .+= sh1d * (wds .* n1)
        crow[nfc .+ cols] .+= sh1d * (wds .* n2)
    end

    # Eliminate the velocity gradient: L_ij = M⁻¹ (E_j λ_i - C_jᵀ u_i)
    MF = cholesky(Symmetric(M))
    MiCx = MF \ Matrix(Cx')
    MiCy = MF \ Matrix(Cy')
    MiE1 = MF \ E1
    MiE2 = MF \ E2

    # Coefficients of L_ij in the momentum equations (volume + edge) and in the
    # flux-continuity rows
    G1 = ν .* (Cx' .- FN1)
    G2 = ν .* (Cy' .- FN2)
    Avisc = -(G1 * MiCx .+ G2 * MiCy)
    Bvisc = G1 * MiE1 .+ G2 * MiE2
    Hvisc = ν .* (HN1 * MiCx .+ HN2 * MiCy)    # flux carries -ν L·n
    Hλvisc = -ν .* (HN1 * MiE1 .+ HN2 * MiE2)

    # Local system for x = [u1; u2; p]
    iu1, iu2, ip = 1:npl, npl .+ (1:npl), 2npl .+ (1:npl)
    j1, j2 = 1:nfc, nfc .+ (1:nfc)
    A = zeros(3npl, 3npl)
    B = zeros(3npl, 2nfc)
    bρ = zeros(3npl)
    r = zeros(3npl)

    A[iu1, iu1] .= Avisc .- X1 .- Ku .+ dtinv .* M .+ Fτ
    A[iu1, iu2] .= -Y1
    A[iu2, iu1] .= -X2
    A[iu2, iu2] .= Avisc .- Y2 .- Ku .+ dtinv .* M .+ Fτ
    A[iu1, ip] .= -Cx' .+ FN1
    A[iu2, ip] .= -Cy' .+ FN2
    A[ip, iu1] .= -Cx'
    A[ip, iu2] .= -Cy'

    B[iu1, j1] .= Bvisc .+ Bc[1, 1] .- Eτ
    B[iu1, j2] .= Bc[1, 2]
    B[iu2, j1] .= Bc[2, 1]
    B[iu2, j2] .= Bvisc .+ Bc[2, 2] .- Eτ
    B[ip, j1] .= E1
    B[ip, j2] .= E2

    r[iu1] .= rcv[1] .- Ku * um[:, 1]
    r[iu2] .= rcv[2] .- Ku * um[:, 2]
    if fe !== nothing
        r[iu1] .+= M * fe[:, 1]
        r[iu2] .+= M * fe[:, 2]
    end
    if ffun !== nothing
        pg = shap' * dg
        fg = reduce(hcat, ffun(view(pg, g, :)) for g in axes(pg, 1))
        r[iu1] .+= shap * (wjac .* fg[1, :])
        r[iu2] .+= shap * (wjac .* fg[2, :])
    end
    if dtinv != 0 && uolde !== nothing
        r[iu1] .+= dtinv .* (M * uolde[:, 1])
        r[iu2] .+= dtinv .* (M * uolde[:, 2])
    end

    # The continuity equations only determine p up to a constant (the constant
    # test function gives the compatibility condition enforced globally), so the
    # first continuity row is replaced by the mean constraint (p,1)_K = ρ |K|.
    icp = ip[1]
    A[icp, :] .= 0.0
    B[icp, :] .= 0.0
    A[icp, ip] .= shap * wjac
    bρ[icp] = -area

    # Flux-continuity rows ⟨(-νL + pI + λ⊗λ)·n + τ(u - λ), μ⟩
    Hx = zeros(2nfc, 3npl)
    Hλ = zeros(2nfc, 2nfc)
    Hx[j1, iu1] .= Hvisc .+ Hτ
    Hx[j2, iu2] .= Hvisc .+ Hτ
    Hx[j1, ip] .= HN1
    Hx[j2, ip] .= HN2
    Hλ[j1, j1] .= Hλvisc .+ Hλc[1, 1] .- Hλτ
    Hλ[j1, j2] .= Hλc[1, 2]
    Hλ[j2, j1] .= Hλc[2, 1]
    Hλ[j2, j2] .= Hλvisc .+ Hλc[2, 2] .- Hλτ
    rH = vcat(rcf[1], rcf[2])

    # Static condensation of (u, p)
    Z = lu!(A) \ hcat(B, bρ, r)
    HxZ = Hx * Z
    Ke = Hλ .- HxZ[:, 1:2nfc]
    Ge = -HxZ[:, 2nfc + 1]
    re = rH .- HxZ[:, 2nfc + 2]

    return Ke, Ge, re, crow, Z, area
end

"Recover the velocity gradient L_ij = M⁻¹(E_j λ_i - C_jᵀ u_i) of one element."
function hdg_recover_gradient(dg, master, ue, λ1e, λ2e)
    vol = hdg_elem_volume(dg, master)
    E1, E2 = hdg_edge_ops(dg, master)
    MF = cholesky(Symmetric(vol.M))
    L = zeros(size(dg, 1), 4)
    L[:, 1] .= MF \ (E1 * λ1e .- vol.Cx' * ue[:, 1])
    L[:, 2] .= MF \ (E2 * λ1e .- vol.Cy' * ue[:, 1])
    L[:, 3] .= MF \ (E1 * λ2e .- vol.Cx' * ue[:, 2])
    L[:, 4] .= MF \ (E2 * λ2e .- vol.Cy' * ue[:, 2])
    return L
end

"""
    hdg_ns_step(master, mesh, ν, dbc; τ=1.0, source=nothing, u=nothing, Λ=nothing,
                uold=nothing, dtinv=0.0)

Performs one Newton step of the HDG discretization of the 2D incompressible
Navier-Stokes equations, linearized about the velocity `u` (npl×2×nt) and the
velocity trace `Λ` (vector of length 2*nps*nf).

# Arguments
- `ν`: kinematic viscosity
- `dbc`: Dirichlet boundary velocity, called as `dbc(p)` with `p` the node
  coordinates, returning the velocity vector `[g1, g2]`
- `τ`: HDG stabilization parameter (τ ≈ ν/ℓ + |u|)
- `source`: body force; `nothing`, a function `p -> [f1, f2]`, or a nodal array
  (npl×2×nt)
- `uold`, `dtinv`: previous-time-level velocity and 1/Δt for backward Euler
  (use `dtinv = 0` for steady state)

# Returns
Named tuple `(u, gradu, p, Λ, ρ)` with the new velocity (npl×2×nt), velocity
gradient (npl×4×nt, columns L11, L12, L21, L22 with Lij = ∂u_i/∂x_j), pressure
(npl×nt, zero global mean), trace vector, and element mean pressures.
"""
function hdg_ns_step(master, mesh, ν, dbc; τ=1.0, source=nothing, u=nothing, Λ=nothing,
                     uold=nothing, dtinv=0.0)
    nps = mesh.porder + 1
    npl = size(mesh.dgnodes, 1)
    nt = size(mesh.t, 1)
    nf = size(mesh.f, 1)
    nfc = 3nps
    nΛ = 2 * nps * nf
    ndof = nΛ + nt
    elcon = mesh.elcon

    u === nothing && (u = zeros(npl, 2, nt))
    Λ === nothing && (Λ = zeros(nΛ))

    src = source isa AbstractArray ? source : nothing
    ffun = source isa Function ? source : nothing

    Kes = zeros(2nfc, 2nfc, nt)
    Ges = zeros(2nfc, nt)
    res = zeros(2nfc, nt)
    crows = zeros(2nfc, nt)
    Zs = zeros(3npl, 2nfc + 2, nt)
    areas = zeros(nt)

    Threads.@threads for it in 1:nt
        λ1e, λ2e = gather_face_vector(Λ, elcon, it, nps)
        fe = src === nothing ? nothing : view(src, :, :, it)
        uolde = uold === nothing ? nothing : view(uold, :, :, it)
        Ke, Ge, re, crow, Z, area = hdg_ns_elemmat(view(mesh.dgnodes, :, :, it), master,
                                                   ν, τ, view(u, :, :, it), λ1e, λ2e,
                                                   fe, ffun, uolde, dtinv)
        Kes[:, :, it] .= Ke
        Ges[:, it] .= Ge
        res[:, it] .= re
        crows[:, it] .= crow
        Zs[:, :, it] .= Z
        areas[it] = area
    end

    # Dirichlet boundary trace DOFs: λ = P_∂g, the face L2 projection of the data
    isdbc = falses(ndof)
    gvals = zeros(ndof)
    sh1d = @view master.sh1d[:, 1, :]
    for i in 1:nf
        mesh.f[i, 4] >= 0 && continue
        Xq, wds, Tm = boundary_face_quad(mesh, master, i)
        gq = reduce(hcat, dbc(view(Xq, g, :)) for g in axes(Xq, 1))
        for c in 1:2
            gproj = Tm \ (sh1d * (wds .* gq[c, :]))
            for k in 1:nps
                gd = 2 * ((i - 1) * nps + k - 1) + c
                isdbc[gd] = true
                gvals[gd] = gproj[k]
            end
        end
    end

    # Global assembly: flux-continuity rows for the trace, compatibility rows
    # for the mean pressure; element 1's compatibility row is replaced by the
    # zero-mean pressure gauge Σ_K |K| ρ_K = 0.
    II = Int[]
    JJ = Int[]
    VV = Float64[]
    sizehint!(II, nt * (2nfc + 2)^2)
    sizehint!(JJ, nt * (2nfc + 2)^2)
    sizehint!(VV, nt * (2nfc + 2)^2)
    rhs = zeros(ndof)
    gdof = zeros(Int, 2nfc)

    for it in 1:nt
        for s in 1:3, a in 1:nps
            ℓ = (s - 1) * nps + a
            gdof[ℓ] = 2 * elcon[a, s, it] - 1
            gdof[nfc + ℓ] = 2 * elcon[a, s, it]
        end
        for jl in 1:2nfc
            gj = gdof[jl]
            for il in 1:2nfc
                gi = gdof[il]
                isdbc[gi] && continue
                push!(II, gi); push!(JJ, gj); push!(VV, Kes[il, jl, it])
            end
            if it != 1
                push!(II, nΛ + it); push!(JJ, gj); push!(VV, crows[jl, it])
            end
        end
        for il in 1:2nfc
            gi = gdof[il]
            isdbc[gi] && continue
            push!(II, gi); push!(JJ, nΛ + it); push!(VV, Ges[il, it])
            rhs[gi] += res[il, it]
        end
    end
    for it in 1:nt
        push!(II, nΛ + 1); push!(JJ, nΛ + it); push!(VV, areas[it])
    end
    for gd in 1:nΛ
        if isdbc[gd]
            push!(II, gd); push!(JJ, gd); push!(VV, 1.0)
            rhs[gd] = gvals[gd]
        end
    end

    H = sparse(II, JJ, VV, ndof, ndof)
    sol = H \ rhs
    Λn = sol[1:nΛ]
    ρ = sol[nΛ+1:end]

    # Element-by-element recovery of (u, p) from Z, then L from the gradient equation
    un = zeros(npl, 2, nt)
    Ln = zeros(npl, 4, nt)
    pn = zeros(npl, nt)
    Threads.@threads for it in 1:nt
        λ1e, λ2e = gather_face_vector(Λn, elcon, it, nps)
        x = Zs[:, end, it] .- Zs[:, 1:2nfc, it] * vcat(λ1e, λ2e) .- Zs[:, 2nfc + 1, it] .* ρ[it]
        un[:, 1, it] .= x[1:npl]
        un[:, 2, it] .= x[npl+1:2npl]
        pn[:, it] .= x[2npl+1:3npl]
        Ln[:, :, it] .= hdg_recover_gradient(view(mesh.dgnodes, :, :, it), master,
                                             view(un, :, :, it), λ1e, λ2e)
    end

    return (u=un, gradu=Ln, p=pn, Λ=Λn, ρ=ρ)
end

"""
    hdg_ns_solve(master, mesh, ν, dbc; τ=1.0, source=nothing, maxiter=20,
                 tol=1e-10, verbose=true, u0=nothing, Λ0=nothing)

Solves the steady 2D incompressible Navier-Stokes equations with the HDG method
by Newton iteration (the first iteration, started from rest, is a Stokes solve).
See [`hdg_ns_step`](@ref) for the arguments and the returned fields.
"""
function hdg_ns_solve(master, mesh, ν, dbc; τ=1.0, source=nothing, maxiter=20,
                      tol=1e-10, verbose=true, u0=nothing, Λ0=nothing)
    u, Λ = u0, Λ0
    result = nothing
    for iter in 1:maxiter
        result = hdg_ns_step(master, mesh, ν, dbc; τ, source, u, Λ)
        Δ = Λ === nothing ? Inf : norm(result.Λ .- Λ) / max(norm(result.Λ), eps())
        verbose && @info "hdg_ns_solve: Newton iteration $iter, Δλ = $Δ"
        u, Λ = result.u, result.Λ
        Δ < tol && break
    end
    return result
end

# ------------------------------------------------------------------------------
# Scalar HDG advection-diffusion transport with a (DG) velocity field, used for
# the temperature equation of the Boussinesq approximation:
#     ∂θ/∂t + ∇·(uθ) - κΔθ = s
# The element convective velocity is the DG velocity u and the face convective
# velocity is the single-valued HDG trace λ, so the numerical flux
#     f̂ = (λ·n)θ̂ - κ q·n + τθ(θ - θ̂),    q = ∇θ
# is conservative.  The gradient q is eliminated analytically exactly as the
# velocity gradient above, so the local solve is a single npl×npl system.
# ------------------------------------------------------------------------------

function hdg_cd_elemmat(dg, master, κ, τθ, ue, λ1e, λ2e, θolde, dtinv, srce, sfun)
    nps = master.porder + 1
    npl = size(dg, 1)
    nfc = 3nps
    sh1d = @view master.sh1d[:, 1, :]

    vol = hdg_elem_volume(dg, master)
    (; shap, shapx, shapy, M, Cx, Cy, wjac) = vol

    u1g = shap' * ue[:, 1]
    u2g = shap' * ue[:, 2]
    Ku = shapx * Diagonal(u1g) * shap' .+ shapy * Diagonal(u2g) * shap'

    E1 = zeros(npl, nfc)
    E2 = zeros(npl, nfc)
    FN1 = zeros(npl, npl)
    FN2 = zeros(npl, npl)
    Fτ = zeros(npl, npl)
    Bλ = zeros(npl, nfc)    # ⟨(λ·n) θ̂ - τθ θ̂, φ⟩
    HN1 = zeros(nfc, npl)
    HN2 = zeros(nfc, npl)
    Hτ = zeros(nfc, npl)
    Hλf = zeros(nfc, nfc)   # ⟨(λ·n) θ̂ - τθ θ̂, μ⟩

    for s in 1:3
        ed = hdg_elem_edge(dg, master, s)
        (; ps, n1, n2, wds) = ed
        cols = (s - 1) * nps .+ (1:nps)

        λn = (sh1d' * λ1e[cols]) .* n1 .+ (sh1d' * λ2e[cols]) .* n2

        T0 = facemat(sh1d, wds)
        Tn1 = facemat(sh1d, wds .* n1)
        Tn2 = facemat(sh1d, wds .* n2)
        Tλn = facemat(sh1d, wds .* λn)

        E1[ps, cols] .+= Tn1
        E2[ps, cols] .+= Tn2
        FN1[ps, ps] .+= Tn1
        FN2[ps, ps] .+= Tn2
        Fτ[ps, ps] .+= τθ .* T0
        Bλ[ps, cols] .+= Tλn .- τθ .* T0
        HN1[cols, ps] .+= Tn1
        HN2[cols, ps] .+= Tn2
        Hτ[cols, ps] .+= τθ .* T0
        Hλf[cols, cols] .+= Tλn .- τθ .* T0
    end

    # Eliminate the gradient: q_j = M⁻¹ (E_j θ̂ - C_jᵀ θ)
    MF = cholesky(Symmetric(M))
    MiCx = MF \ Matrix(Cx')
    MiCy = MF \ Matrix(Cy')
    MiE1 = MF \ E1
    MiE2 = MF \ E2

    G1 = κ .* (Cx' .- FN1)
    G2 = κ .* (Cy' .- FN2)

    A = dtinv .* M .- Ku .+ Fτ .- (G1 * MiCx .+ G2 * MiCy)
    B = G1 * MiE1 .+ G2 * MiE2 .+ Bλ
    r = zeros(npl)
    if θolde !== nothing && dtinv != 0
        r .+= dtinv .* (M * θolde)
    end
    if srce !== nothing
        r .+= M * srce
    end
    if sfun !== nothing
        pg = shap' * dg
        r .+= shap * (wjac .* [sfun(view(pg, g, :)) for g in axes(pg, 1)])
    end

    Hx = κ .* (HN1 * MiCx .+ HN2 * MiCy) .+ Hτ           # flux carries -κ q·n
    Hλ = -κ .* (HN1 * MiE1 .+ HN2 * MiE2) .+ Hλf

    Z = lu!(A) \ hcat(B, r)
    HxZ = Hx * Z
    Ke = Hλ .- HxZ[:, 1:nfc]
    re = -HxZ[:, nfc + 1]

    return Ke, re, Z
end

"Recover the scalar gradient q_j = M⁻¹(E_j θ̂ - C_jᵀ θ) of one element."
function hdg_recover_scalargrad(dg, master, θe, θ̂e)
    vol = hdg_elem_volume(dg, master)
    E1, E2 = hdg_edge_ops(dg, master)
    MF = cholesky(Symmetric(vol.M))
    q = zeros(size(dg, 1), 2)
    q[:, 1] .= MF \ (E1 * θ̂e .- vol.Cx' * θe)
    q[:, 2] .= MF \ (E2 * θ̂e .- vol.Cy' * θe)
    return q
end

"""
    hdg_cd_step(master, mesh, κ, tbc; τ=1.0, u=nothing, Λ=nothing, θold=nothing,
                dtinv=0.0, source=nothing)

Solves one (linear) implicit step of the scalar HDG advection-diffusion
equation ∂θ/∂t + ∇·(uθ) - κΔθ = s with the DG velocity field `u` (npl×2×nt)
and the velocity trace `Λ` from the HDG Navier-Stokes solver.

`tbc(p, tag)` prescribes the boundary condition on a boundary face with tag
`tag` (`-mesh.f[i,4]`) at node coordinates `p`: return `(:d, value)` for a
Dirichlet condition on θ̂ or `(:n, flux)` for a prescribed total normal flux
(e.g. `(:n, 0.0)` for an insulated wall).

Returns a named tuple `(θ, q, Θ)` with the scalar field (npl×nt), its gradient
q = ∇θ (npl×2×nt), and the face trace.
"""
function hdg_cd_step(master, mesh, κ, tbc; τ=1.0, u=nothing, Λ=nothing, θold=nothing,
                     dtinv=0.0, source=nothing)
    nps = mesh.porder + 1
    npl = size(mesh.dgnodes, 1)
    nt = size(mesh.t, 1)
    nf = size(mesh.f, 1)
    nfc = 3nps
    ndof = nps * nf
    elcon = mesh.elcon

    u === nothing && (u = zeros(npl, 2, nt))
    Λ === nothing && (Λ = zeros(2 * nps * nf))

    src = source isa AbstractArray ? source : nothing
    sfun = source isa Function ? source : nothing

    Kes = zeros(nfc, nfc, nt)
    res = zeros(nfc, nt)
    Zs = zeros(npl, nfc + 1, nt)

    Threads.@threads for it in 1:nt
        λ1e, λ2e = gather_face_vector(Λ, elcon, it, nps)
        θolde = θold === nothing ? nothing : view(θold, :, it)
        srce = src === nothing ? nothing : view(src, :, it)
        Ke, re, Z = hdg_cd_elemmat(view(mesh.dgnodes, :, :, it), master, κ, τ,
                                   view(u, :, :, it), λ1e, λ2e, θolde, dtinv, srce, sfun)
        Kes[:, :, it] .= Ke
        res[:, it] .= re
        Zs[:, :, it] .= Z
    end

    # Boundary conditions: Dirichlet data enters through its face L2 projection
    # (replacing the trace rows); prescribed-flux faces keep their
    # flux-continuity row with ⟨g_N, μ⟩ on the right-hand side.
    isdbc = falses(ndof)
    gvals = zeros(ndof)
    nrhs = zeros(ndof)
    sh1d = @view master.sh1d[:, 1, :]
    for i in 1:nf
        mesh.f[i, 4] >= 0 && continue
        tag = -mesh.f[i, 4]
        Xq, wds, Tm = boundary_face_quad(mesh, master, i)
        bc = [tbc(view(Xq, g, :), tag) for g in axes(Xq, 1)]
        vals = Float64[b[2] for b in bc]
        if bc[1][1] == :d
            gproj = Tm \ (sh1d * (wds .* vals))
            for k in 1:nps
                gd = (i - 1) * nps + k
                isdbc[gd] = true
                gvals[gd] = gproj[k]
            end
        else
            nrhs[(i - 1) * nps .+ (1:nps)] .+= sh1d * (wds .* vals)
        end
    end

    II = Int[]
    JJ = Int[]
    VV = Float64[]
    rhs = zeros(ndof)
    gdof = zeros(Int, nfc)

    for it in 1:nt
        for s in 1:3, a in 1:nps
            gdof[(s - 1) * nps + a] = elcon[a, s, it]
        end
        for jl in 1:nfc
            gj = gdof[jl]
            for il in 1:nfc
                gi = gdof[il]
                isdbc[gi] && continue
                push!(II, gi); push!(JJ, gj); push!(VV, Kes[il, jl, it])
            end
        end
        for il in 1:nfc
            gi = gdof[il]
            isdbc[gi] || (rhs[gi] += res[il, it])
        end
    end
    rhs .+= nrhs
    for gd in 1:ndof
        if isdbc[gd]
            push!(II, gd); push!(JJ, gd); push!(VV, 1.0)
            rhs[gd] = gvals[gd]
        end
    end

    H = sparse(II, JJ, VV, ndof, ndof)
    Θ = H \ rhs

    θn = zeros(npl, nt)
    qn = zeros(npl, 2, nt)
    Threads.@threads for it in 1:nt
        θ̂e = gather_face_scalar(Θ, elcon, it, nps)
        θn[:, it] .= Zs[:, end, it] .- Zs[:, 1:nfc, it] * θ̂e
        qn[:, :, it] .= hdg_recover_scalargrad(view(mesh.dgnodes, :, :, it), master,
                                               view(θn, :, it), θ̂e)
    end

    return (θ=θn, q=qn, Θ=Θ)
end

# ------------------------------------------------------------------------------
# Divergence-free postprocessing (paper §3.3, two-dimensional version)
# ------------------------------------------------------------------------------

"""
    hdg_ns_postprocess(master, mesh, master1, mesh1, result)

Element-by-element postprocessing of the HDG Navier-Stokes solution (paper
§3.3): finds u* ∈ [P_{k+1}(K)]² such that, on every edge F of K,

    ⟨(u* - û)·n, μ⟩_F = 0                          ∀μ ∈ P_k(F),
    ⟨t·∇(u*·n) - nᵀ{{L}}t, t·∇μ⟩_F = 0             ∀μ ∈ P_{k+1}(F)^⊥,

and, in the element interior,

    (u* - u, ∇w)_K = 0                             ∀w ∈ P_k(K),
    (∇×u* - w_h, w b_K)_K = 0                      ∀w ∈ P_{k-1}(K),

where {{L}} is the single-valued average of the velocity gradient across the
face, w_h = L21 - L12 the discrete vorticity, and b_K the cubic bubble.  The
resulting velocity is exactly divergence-free, H(div)-conforming, and converges
with order k+2 for k ≥ 1.

`master1`/`mesh1` hold the degree k+1 approximation on the same triangulation
and must be built with the same quadrature order as `master`/`mesh` (see
`hdg_postprocess` for the same convention).  Returns u* as a (npl1×2×nt) array
on the nodes of `mesh1`.
"""
function hdg_ns_postprocess(master, mesh, master1, mesh1, result)
    porder = master.porder
    @assert porder >= 1 "postprocessing requires k ≥ 1"
    @assert length(master.gwgh) == length(master1.gwgh) "master and master1 must use the same quadrature"
    nps = porder + 1
    npl = size(mesh.dgnodes, 1)
    npl1 = size(mesh1.dgnodes, 1)
    nt = size(mesh.t, 1)
    nk1 = porder * (porder + 1) ÷ 2          # dim P_{k-1}
    u, L, Λ = result.u, result.gradu, result.Λ
    elcon = mesh.elcon

    sh1d0 = master.sh1d[:, 1, :]
    sh1d1 = master1.sh1d[:, 1, :]
    sh1d1x = master1.sh1d[:, 2, :]
    gw1d = master.gw1d
    shap1 = master1.shap[:, 1, :]

    # the single P_{k+1}(F)^⊥ test function: orthogonalize the top Koornwinder
    # mode against P_k(F) in the reference inner product (exact for straight
    # edges; its tangential derivative is evaluated through the edge metric)
    f1d, fx1d = koornwinder1d(master.gp1d, porder + 1)

    # P_{k-1} modes and the cubic bubble at the volume quadrature points
    Wd, _, _ = koornwinder2d(master1.gpts, porder - 1)
    ξ, η = master1.gpts[:, 1], master1.gpts[:, 2]
    bk = ξ .* η .* (1 .- ξ .- η)

    nrow = 3nps + 3 + npl + nk1
    ra, rb, rc, rd = 0, 3nps, 3nps + 3, 3nps + 3 + npl

    ustar = zeros(npl1, 2, nt)
    Threads.@threads for it in 1:nt
        dg0 = view(mesh.dgnodes, :, :, it)
        dg1 = view(mesh1.dgnodes, :, :, it)
        vol0 = hdg_elem_volume(dg0, master)
        vol1 = hdg_elem_volume(dg1, master1)
        λ1e, λ2e = gather_face_vector(Λ, elcon, it, nps)

        E = zeros(nrow, 2npl1)
        b = zeros(nrow)
        cu1 = 1:npl1
        cu2 = npl1 .+ (1:npl1)

        for s in 1:3
            ed = hdg_elem_edge(dg1, master1, s)
            (; n1, n2, ds, wds) = ed
            ip1 = master1.perm[:, s, 1]
            cols = (s - 1) * nps .+ (1:nps)

            # ⟨(u* - û)·n, μ⟩_F = 0, μ ∈ P_k(F)
            rows = ra .+ cols
            E[rows, cu1[ip1]] .+= sh1d0 * Diagonal(wds .* n1) * sh1d1'
            E[rows, cu2[ip1]] .+= sh1d0 * Diagonal(wds .* n2) * sh1d1'
            λn = (sh1d0' * λ1e[cols]) .* n1 .+ (sh1d0' * λ2e[cols]) .* n2
            b[rows] .= sh1d0 * (wds .* λn)

            # ⟨t·∇(u*·n) - nᵀ{{L}}t, t·∇μ⟩_F = 0, μ ∈ P_{k+1}(F)^⊥
            G = f1d' * Diagonal(wds) * f1d
            c = G \ [zeros(porder + 1); 1.0]
            μx = fx1d * c
            row = rb + s
            E[row, cu1[ip1]] .+= sh1d1x * (gw1d .* μx .* n1 ./ ds)
            E[row, cu2[ip1]] .+= sh1d1x * (gw1d .* μx .* n2 ./ ds)

            # face-average of nᵀ L t with the tangent t = (-n2, n1)
            a = nLt_edge(L, it, s, master, n1, n2)
            i_f = abs(mesh.t2f[it, s])
            jt = mesh.f[i_f, 3] == it ? mesh.f[i_f, 4] : mesh.f[i_f, 3]
            if jt > 0
                s2 = findfirst(x -> abs(x) == i_f, mesh.t2f[jt, :])
                a = (a .+ nLt_edge(L, jt, s2, master, n1, n2; rev=true)) ./ 2
            end
            b[row] = sum(gw1d .* μx .* a)
        end

        # (u* - u, ∇w)_K = 0, w ∈ P_k(K)
        rows = rc .+ (1:npl)
        E[rows, cu1] .= vol0.shapx * shap1'
        E[rows, cu2] .= vol0.shapy * shap1'
        b[rows] .= vol0.shapx * (vol0.shap' * u[:, 1, it]) .+
                   vol0.shapy * (vol0.shap' * u[:, 2, it])

        # (∇×u* - w_h, w b_K)_K = 0, w ∈ P_{k-1}(K)
        Ψ = Wd .* bk
        rows = rd .+ (1:nk1)
        E[rows, cu1] .= -Ψ' * vol1.shapy'
        E[rows, cu2] .= Ψ' * vol1.shapx'
        whg = vol0.shap' * (L[:, 3, it] .- L[:, 2, it])
        b[rows] .= Ψ' * (vol1.wjac .* whg)

        # one consistent redundancy (constant w above) -> least-squares solve
        x = E \ b
        ustar[:, 1, it] .= x[cu1]
        ustar[:, 2, it] .= x[cu2]
    end

    return ustar
end

# trace of nᵀ L t (t = (-n2, n1)) of element `it` on its local edge `s`,
# evaluated at the 1D quadrature points; `rev` reverses the nodal values to
# match the orientation of the neighboring element across the face
function nLt_edge(L, it, s, master, n1, n2; rev=false)
    sh1d0 = @view master.sh1d[:, 1, :]
    ps = master.perm[:, s, 1]
    t1, t2 = -n2, n1
    Lg = ntuple(c -> sh1d0' * (rev ? reverse(L[ps, c, it]) : L[ps, c, it]), 4)
    return n1 .* (Lg[1] .* t1 .+ Lg[2] .* t2) .+ n2 .* (Lg[3] .* t1 .+ Lg[4] .* t2)
end
