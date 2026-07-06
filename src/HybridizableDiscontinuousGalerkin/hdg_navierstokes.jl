using LinearAlgebra
using SparseArrays
using StaticArrays
using TwoDG.Masters: koornwinder1d, koornwinder2d
using TwoDG.Geometry: face_normal, adjugate

# HDG method for the incompressible Navier-Stokes equations following
# Nguyen, Peraire & Cockburn, "An implicit high-order hybridizable discontinuous
# Galerkin method for the incompressible Navier-Stokes equations", JCP 230 (2011).
# Dimension-generic (THREED_PLAN Phase E3): the same element assembly serves
# triangles (Dim = 2) and tetrahedra (Dim = 3); the spatial direction is an
# array axis and the velocity has Dim components.
#
# The velocity gradient - pressure - velocity formulation is discretized:
#     L - ∇u = 0
#     ∂u/∂t - ν∇·L + ∇p + ∇·(u⊗u) = f
#     ∇·u = 0
# with the single-valued numerical trace û = λ on faces and total-stress flux
#     f̂ = (-νL + pI + λ⊗λ)·n + τ(u - λ).
# Following solution strategy A of the paper (§4.2), the velocity gradient is
# eliminated element-by-element through the gradient equation
#     L_id = M⁻¹ (E_d λ_i - C_dᵀ u_i),
# the remaining (u, p) local system is statically condensed, and the globally
# coupled unknowns are the trace of the velocity λ and the mean of the pressure
# ρ_K on each element, giving the saddle-point system
#     [A Bᵀ; B 0] [λ; ρ] = [R; 0]
# closed with the global zero-mean pressure gauge Σ_K |K| ρ_K = 0.
#
# Conventions: Lij = ∂u_i/∂x_j.  Element-local trace DOFs live in the
# face-scalar space of the (Dim+1)·nps face nodes (local face s, face node a
# -> index (s-1)*nps + a), one block per velocity component; globally, face f,
# face node k, and component c give the interleaved index
# Dim*((f-1)*nps + k - 1) + c.

# ------------------------------------------------------------------------------
# Shared element machinery
# ------------------------------------------------------------------------------

"""
    hdg_elem_volume(dg, master)

Volume metric terms of one (possibly curved) element with nodes `dg`
(npl × Dim): shape values `shap` (npl × ng), the quadrature- and
Jacobian-weighted physical derivative tables `shapd` (npl × ng × Dim,
`shapd[m, g, d] = w_g jac_g ∂φ_m/∂x_d`), the mass matrix `M`, the
convection-type matrices `C` (npl × npl × Dim, `C[m, n, d] = (φ_m, ∂_d φ_n)`),
and the weighted Jacobian `wjac`.
"""
function hdg_elem_volume(dg, master)
    Dim = ndims(master)
    shap = @view master.shap[:, 1, :]
    gw = master.gwgh
    npl, ng = size(shap, 1), size(shap, 2)

    shapd = zeros(npl, ng, Dim)
    wjac = zeros(ng)
    for g in 1:ng
        # J[k, d] = ∂x_d/∂ξ_k; the adjugate carries det(J) ∂ξ_k/∂x_d, so the
        # weighted tables absorb det(J) without a division
        J = SMatrix{Dim, Dim}(ntuple(Val(Dim * Dim)) do idx
            k = (idx - 1) % Dim + 1
            d = (idx - 1) ÷ Dim + 1
            s = 0.0
            @inbounds for m in 1:npl
                s += master.shap[m, 1 + k, g] * dg[m, d]
            end
            s
        end)
        adj = adjugate(J)
        wjac[g] = gw[g] * det(J)
        @inbounds for d in 1:Dim, m in 1:npl
            acc = 0.0
            for k in 1:Dim
                acc += master.shap[m, 1 + k, g] * adj[d, k]
            end
            shapd[m, g, d] = gw[g] * acc
        end
    end

    M = shap * Diagonal(wjac) * shap'
    C = zeros(npl, npl, Dim)
    for d in 1:Dim
        C[:, :, d] .= shap * @view(shapd[:, :, d])'
    end

    return (; shap, shapd, M, C, wjac)
end

"""
    hdg_elem_face(dg, master, s)

Face metric terms of local face `s` of the element with nodes `dg`: the volume
node indices `ps` on the face (canonical traversal), the outward unit normal
`n` (ngf × Dim), the face measure Jacobian `ds`, and the weighted measure
`wds` at the face quadrature points. The normal comes from `face_normal`
dispatch (tangent rotation in 2D, cross product in 3D).
"""
function hdg_elem_face(dg, master, s)
    Dim = ndims(master)
    fe = master.face
    ps = master.perm[:, s, 1]
    coords = dg[ps, :]
    ngf = size(fe.shap, 3)

    n = zeros(ngf, Dim)
    ds = zeros(ngf)
    for g in 1:ngf
        τ = ntuple(Val(Dim - 1)) do k
            SVector{Dim}(ntuple(Val(Dim)) do d
                acc = 0.0
                @inbounds for m in axes(coords, 1)
                    acc += fe.shap[m, 1 + k, g] * coords[m, d]
                end
                acc
            end)
        end
        nvec = face_normal(τ...)
        nn = norm(nvec)
        for d in 1:Dim
            n[g, d] = nvec[d] / nn
        end
        ds[g] = nn
    end
    wds = fe.gwgh .* ds
    return (; ps, n, ds, wds)
end

"Face matrix `⟨w μ_a, μ_b⟩_F` in the nps face nodes for a quadrature weight `w`."
facemat(shf, w) = shf * Diagonal(w) * shf'

"""
    hdg_face_ops(dg, master)

The lifting operators `E` (npl × nfc × Dim) of the gradient equation,
`E[m, ℓ, d] = ⟨μ_ℓ, φ_m n_d⟩_∂K`, mapping face-scalar trace values to volume
test functions.
"""
function hdg_face_ops(dg, master)
    Dim = ndims(master)
    nfe = Dim + 1
    nps = size(master.perm, 1)
    npl = size(dg, 1)
    shf = @view master.face.shap[:, 1, :]
    E = zeros(npl, nfe * nps, Dim)
    for s in 1:nfe
        fc = hdg_elem_face(dg, master, s)
        cols = (s - 1) * nps .+ (1:nps)
        for d in 1:Dim
            E[fc.ps, cols, d] .+= facemat(shf, fc.wds .* @view(fc.n[:, d]))
        end
    end
    return E
end

"Element-local face-scalar values (nfc) of a face field `Θ` (one DOF per face node)."
function gather_face_scalar(Θ, elcon, it, nps)
    nfe = size(elcon, 2)
    v = zeros(nfe * nps)
    for s in 1:nfe, a in 1:nps
        v[(s - 1) * nps + a] = Θ[elcon[a, s, it]]
    end
    return v
end

"Element-local face-scalar components (nfc × Dim) of the interleaved trace vector `Λ`."
function gather_face_vector(Λ, elcon, it, nps, Dim)
    nfe = size(elcon, 2)
    v = zeros(nfe * nps, Dim)
    for s in 1:nfe, a in 1:nps
        g = elcon[a, s, it]
        for c in 1:Dim
            v[(s - 1) * nps + a, c] = Λ[Dim * (g - 1) + c]
        end
    end
    return v
end

"""
    boundary_face_quad(mesh, master, i)

Quadrature data of boundary face `i` in global face-node order: the physical
quadrature points `Xq` (ngf × Dim), the weighted measure `wds`, and the face
mass matrix `Tm`.  Used to impose boundary data through its L2 projection
P_∂g onto P_k(F) — nodal interpolation of the data would introduce an
O(h^{k+1}) perturbation that destroys the superconvergence of the method.
"""
function boundary_face_quad(mesh, master, i)
    Dim = ndims(master)
    it = mesh.f[i, Dim + 1]
    lf = findfirst(==(i), mesh.t2f[it, :])
    ori = mesh.t2o[it, lf]
    pp = master.perm[:, lf, ori]
    coords = mesh.dgnodes[pp, :, it]

    fe = master.face
    shf = @view fe.shap[:, 1, :]
    ngf = size(fe.shap, 3)
    wds = zeros(ngf)
    for g in 1:ngf
        τ = ntuple(Val(Dim - 1)) do k
            SVector{Dim}(ntuple(Val(Dim)) do d
                acc = 0.0
                @inbounds for m in axes(coords, 1)
                    acc += fe.shap[m, 1 + k, g] * coords[m, d]
                end
                acc
            end)
        end
        wds[g] = fe.gwgh[g] * norm(face_normal(τ...))
    end
    Xq = shf' * coords
    Tm = shf * Diagonal(wds) * shf'
    return Xq, wds, Tm
end

# ------------------------------------------------------------------------------
# Incompressible Navier-Stokes
# ------------------------------------------------------------------------------

"""
    hdg_ns_elemmat(dg, master, ν, τ, um, λe, fe, ffun, uolde, dtinv)

Element matrices for one Newton step of the HDG incompressible Navier-Stokes
discretization, linearized about the velocity `um` (npl × Dim) and the trace
components `λe` (nfc × Dim).  The body force is either the DG field `fe`
(npl × Dim, integrated exactly) or the function `ffun` (evaluated at the
quadrature points — interpolating it at the nodes instead would spoil the
superconvergence of the method); `uolde` is the velocity at the previous time
level for backward Euler (or nothing), and `dtinv = 1/Δt` (0 for steady state).

The velocity gradient is eliminated analytically and the local (u, p) system is
statically condensed.  Returns `(Ke, Ge, re, crow, Z, area)`: the condensed
trace matrix (Dim·nfc × Dim·nfc, trace ordered [λ₁; …; λ_Dim]), the
mean-pressure coupling column, the condensed right-hand side, the element-
compatibility row ⟨λ·n, 1⟩_∂K, the local solution operator `Z = A⁻¹[B  bρ  r]`
for the recovery of (u, p), and the element measure.
"""
function hdg_ns_elemmat(dg, master, ν, τ, um, λe, fe, ffun, uolde, dtinv)
    Dim = ndims(master)
    nfe = Dim + 1
    nps = size(master.perm, 1)
    npl = size(dg, 1)
    nfc = nfe * nps
    shf = @view master.face.shap[:, 1, :]

    vol = hdg_elem_volume(dg, master)
    (; shap, shapd, M, C, wjac) = vol
    area = sum(wjac)

    # Newton linearization of the volume convection about um:
    # XX[:, :, d, c] = shapd_d Diag(u_c at quad) shap', Ku = Σ_d XX[:, :, d, d]
    ug = shap' * um                                     # (ng, Dim)
    XX = zeros(npl, npl, Dim, Dim)
    for c in 1:Dim, d in 1:Dim
        XX[:, :, d, c] .= @view(shapd[:, :, d]) * Diagonal(@view(ug[:, c])) * shap'
    end
    Ku = copy(XX[:, :, 1, 1])
    for d in 2:Dim
        Ku .+= @view XX[:, :, d, d]
    end

    # Face operators in the face-scalar space
    E = zeros(npl, nfc, Dim)         # ⟨λ, φ n_d⟩_∂K (gradient lifting / continuity)
    FN = zeros(npl, npl, Dim)        # ⟨(volume field) n_d, φ⟩_∂K
    Fτ = zeros(npl, npl)             # τ ⟨u, φ⟩_∂K
    Eτ = zeros(npl, nfc)             # τ ⟨λ, φ⟩_∂K
    Bc = zeros(npl, nfc, Dim, Dim)   # linearized ⟨(λ·n) λ_i, φ⟩ in [:, :, i, j]
    rcv = zeros(npl, Dim)            # its Newton constant term
    Hλc = zeros(nfc, nfc, Dim, Dim)  # the same, tested on faces
    rcf = zeros(nfc, Dim)
    HN = zeros(nfc, npl, Dim)        # ⟨(volume field) n_d, μ⟩_∂K
    Hτ = zeros(nfc, npl)             # τ ⟨u, μ⟩_∂K
    Hλτ = zeros(nfc, nfc)            # τ ⟨λ, μ⟩_∂K
    crow = zeros(Dim * nfc)          # ⟨λ·n, 1⟩_∂K

    for s in 1:nfe
        fc = hdg_elem_face(dg, master, s)
        (; ps, n, wds) = fc
        cols = (s - 1) * nps .+ (1:nps)

        λg = shf' * λe[cols, :]                          # (ngf, Dim)
        λn = @view(λg[:, 1]) .* @view(n[:, 1])
        for d in 2:Dim
            λn = λn .+ @view(λg[:, d]) .* @view(n[:, d])
        end

        T0 = facemat(shf, wds)
        Tλn = facemat(shf, wds .* λn)

        for d in 1:Dim
            Tn = facemat(shf, wds .* @view(n[:, d]))
            E[ps, cols, d] .+= Tn
            FN[ps, ps, d] .+= Tn
            HN[cols, ps, d] .+= Tn
        end
        Fτ[ps, ps] .+= τ .* T0
        Eτ[ps, cols] .+= τ .* T0
        Hτ[cols, ps] .+= τ .* T0
        Hλτ[cols, cols] .+= τ .* T0

        # Newton linearization of the trace convection (λ·n)λ_i:
        # (λᵐ·n) δλ_i + (δλ·n) λ_iᵐ, with constant term (λᵐ·n) λ_iᵐ
        for i in 1:Dim, j in 1:Dim
            Tij = facemat(shf, wds .* @view(n[:, j]) .* @view(λg[:, i]))
            i == j && (Tij .+= Tλn)
            Bc[ps, cols, i, j] .+= Tij
            Hλc[cols, cols, i, j] .+= Tij
        end
        for i in 1:Dim
            rc = shf * (wds .* λn .* @view(λg[:, i]))
            rcv[ps, i] .+= rc
            rcf[cols, i] .+= rc
        end
        for d in 1:Dim
            crow[(d - 1) * nfc .+ cols] .+= shf * (wds .* @view(n[:, d]))
        end
    end

    # Eliminate the velocity gradient: L_id = M⁻¹ (E_d λ_i - C_dᵀ u_i)
    MF = cholesky(Symmetric(M))
    MiC = zeros(npl, npl, Dim)
    MiE = zeros(npl, nfc, Dim)
    for d in 1:Dim
        MiC[:, :, d] .= MF \ Matrix(@view(C[:, :, d])')
        MiE[:, :, d] .= MF \ @view(E[:, :, d])
    end

    # Coefficients of L in the momentum equations (volume + face) and in the
    # flux-continuity rows
    Avisc = zeros(npl, npl)
    Bvisc = zeros(npl, nfc)
    Hvisc = zeros(nfc, npl)
    Hλvisc = zeros(nfc, nfc)
    for d in 1:Dim
        Gd = ν .* (@view(C[:, :, d])' .- @view(FN[:, :, d]))
        Avisc .-= Gd * @view(MiC[:, :, d])
        Bvisc .+= Gd * @view(MiE[:, :, d])
        Hvisc .+= ν .* (@view(HN[:, :, d]) * @view(MiC[:, :, d]))    # flux carries -ν L·n
        Hλvisc .-= ν .* (@view(HN[:, :, d]) * @view(MiE[:, :, d]))
    end

    # Local system for x = [u_1; …; u_Dim; p]
    nv = (Dim + 1) * npl
    iu = ntuple(c -> (c - 1) * npl .+ (1:npl), Dim)
    ip = Dim * npl .+ (1:npl)
    jλ = ntuple(c -> (c - 1) * nfc .+ (1:nfc), Dim)
    A = zeros(nv, nv)
    B = zeros(nv, Dim * nfc)
    bρ = zeros(nv)
    r = zeros(nv)

    for ci in 1:Dim
        A[iu[ci], iu[ci]] .= Avisc .- Ku .+ dtinv .* M .+ Fτ
        for cj in 1:Dim
            A[iu[ci], iu[cj]] .-= @view XX[:, :, cj, ci]
        end
        A[iu[ci], ip] .= .-@view(C[:, :, ci])' .+ @view(FN[:, :, ci])
        A[ip, iu[ci]] .= .-@view(C[:, :, ci])'

        B[iu[ci], jλ[ci]] .= Bvisc .- Eτ
        for cj in 1:Dim
            B[iu[ci], jλ[cj]] .+= @view Bc[:, :, ci, cj]
        end
        B[ip, jλ[ci]] .= @view E[:, :, ci]

        r[iu[ci]] .= @view(rcv[:, ci]) .- Ku * @view(um[:, ci])
        if fe !== nothing
            r[iu[ci]] .+= M * @view(fe[:, ci])
        end
        if dtinv != 0 && uolde !== nothing
            r[iu[ci]] .+= dtinv .* (M * @view(uolde[:, ci]))
        end
    end
    if ffun !== nothing
        pg = shap' * dg
        fg = reduce(hcat, ffun(view(pg, g, :)) for g in axes(pg, 1))   # (Dim, ng)
        for ci in 1:Dim
            r[iu[ci]] .+= shap * (wjac .* @view(fg[ci, :]))
        end
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
    Hx = zeros(Dim * nfc, nv)
    Hλ = zeros(Dim * nfc, Dim * nfc)
    for ci in 1:Dim
        Hx[jλ[ci], iu[ci]] .= Hvisc .+ Hτ
        Hx[jλ[ci], ip] .= @view HN[:, :, ci]
        Hλ[jλ[ci], jλ[ci]] .= Hλvisc .- Hλτ
        for cj in 1:Dim
            Hλ[jλ[ci], jλ[cj]] .+= @view Hλc[:, :, ci, cj]
        end
    end
    rH = vec(rcf)

    # Static condensation of (u, p)
    Z = lu!(A) \ hcat(B, bρ, r)
    HxZ = Hx * Z
    Ke = Hλ .- HxZ[:, 1:(Dim * nfc)]
    Ge = -HxZ[:, Dim * nfc + 1]
    re = rH .- HxZ[:, Dim * nfc + 2]

    return Ke, Ge, re, crow, Z, area
end

"""
Recover the velocity gradient L_ij = M⁻¹(E_j λ_i - C_jᵀ u_i) of one element;
column layout `(i - 1) Dim + j` (L11, L12, …, L1Dim, L21, …).
"""
function hdg_recover_gradient(dg, master, ue, λe)
    Dim = ndims(master)
    vol = hdg_elem_volume(dg, master)
    E = hdg_face_ops(dg, master)
    MF = cholesky(Symmetric(vol.M))
    L = zeros(size(dg, 1), Dim * Dim)
    for i in 1:Dim, j in 1:Dim
        L[:, (i - 1) * Dim + j] .= MF \ (@view(E[:, :, j]) * @view(λe[:, i]) .-
                                         @view(vol.C[:, :, j])' * @view(ue[:, i]))
    end
    return L
end

"""
    hdg_ns_step(master, mesh, ν, dbc; τ=1.0, source=nothing, u=nothing, Λ=nothing,
                uold=nothing, dtinv=0.0)

Performs one Newton step of the HDG discretization of the incompressible
Navier-Stokes equations (2D or 3D, from the mesh), linearized about the
velocity `u` (npl × Dim × nt) and the velocity trace `Λ` (vector of length
Dim·nps·nf).

# Arguments
- `ν`: kinematic viscosity
- `dbc`: Dirichlet boundary velocity, called as `dbc(p)` with `p` the node
  coordinates, returning the velocity vector `[g1, …, gDim]`
- `τ`: HDG stabilization parameter (τ ≈ ν/ℓ + |u|)
- `source`: body force; `nothing`, a function `p -> [f1, …, fDim]`, or a nodal
  array (npl × Dim × nt)
- `uold`, `dtinv`: previous-time-level velocity and 1/Δt for backward Euler
  (use `dtinv = 0` for steady state)

# Returns
Named tuple `(u, gradu, p, Λ, ρ)` with the new velocity (npl × Dim × nt),
velocity gradient (npl × Dim² × nt, column `(i-1)Dim + j` holding
Lij = ∂u_i/∂x_j), pressure (npl × nt, zero global mean), trace vector, and
element mean pressures.
"""
function hdg_ns_step(master, mesh, ν, dbc; τ=1.0, source=nothing, u=nothing, Λ=nothing,
                     uold=nothing, dtinv=0.0)
    Dim = ndims(master)
    nfe = Dim + 1
    nps = size(master.perm, 1)
    npl = size(mesh.dgnodes, 1)
    nt = size(mesh.t, 1)
    nf = size(mesh.f, 1)
    nfc = nfe * nps
    nΛ = Dim * nps * nf
    ndof = nΛ + nt
    elcon = mesh.elcon

    u === nothing && (u = zeros(npl, Dim, nt))
    Λ === nothing && (Λ = zeros(nΛ))

    src = source isa AbstractArray ? source : nothing
    ffun = source isa Function ? source : nothing

    Kes = zeros(Dim * nfc, Dim * nfc, nt)
    Ges = zeros(Dim * nfc, nt)
    res = zeros(Dim * nfc, nt)
    crows = zeros(Dim * nfc, nt)
    Zs = zeros((Dim + 1) * npl, Dim * nfc + 2, nt)
    areas = zeros(nt)

    Threads.@threads for it in 1:nt
        λe = gather_face_vector(Λ, elcon, it, nps, Dim)
        fe = src === nothing ? nothing : view(src, :, :, it)
        uolde = uold === nothing ? nothing : view(uold, :, :, it)
        Ke, Ge, re, crow, Z, area = hdg_ns_elemmat(view(mesh.dgnodes, :, :, it), master,
                                                   ν, τ, view(u, :, :, it), λe,
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
    shf = @view master.face.shap[:, 1, :]
    for i in 1:nf
        mesh.f[i, end] >= 0 && continue
        Xq, wds, Tm = boundary_face_quad(mesh, master, i)
        gq = reduce(hcat, dbc(view(Xq, g, :)) for g in axes(Xq, 1))
        for c in 1:Dim
            gproj = Tm \ (shf * (wds .* gq[c, :]))
            for k in 1:nps
                gd = Dim * ((i - 1) * nps + k - 1) + c
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
    sizehint!(II, nt * (Dim * nfc + 2)^2)
    sizehint!(JJ, nt * (Dim * nfc + 2)^2)
    sizehint!(VV, nt * (Dim * nfc + 2)^2)
    rhs = zeros(ndof)
    gdof = zeros(Int, Dim * nfc)

    for it in 1:nt
        for s in 1:nfe, a in 1:nps
            ℓ = (s - 1) * nps + a
            g = elcon[a, s, it]
            for c in 1:Dim
                gdof[(c - 1) * nfc + ℓ] = Dim * (g - 1) + c
            end
        end
        for jl in 1:(Dim * nfc)
            gj = gdof[jl]
            for il in 1:(Dim * nfc)
                gi = gdof[il]
                isdbc[gi] && continue
                push!(II, gi); push!(JJ, gj); push!(VV, Kes[il, jl, it])
            end
            if it != 1
                push!(II, nΛ + it); push!(JJ, gj); push!(VV, crows[jl, it])
            end
        end
        for il in 1:(Dim * nfc)
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
    un = zeros(npl, Dim, nt)
    Ln = zeros(npl, Dim * Dim, nt)
    pn = zeros(npl, nt)
    Threads.@threads for it in 1:nt
        λe = gather_face_vector(Λn, elcon, it, nps, Dim)
        x = Zs[:, end, it] .- Zs[:, 1:(Dim * nfc), it] * vec(λe) .-
            Zs[:, Dim * nfc + 1, it] .* ρ[it]
        for c in 1:Dim
            un[:, c, it] .= x[(c - 1) * npl .+ (1:npl)]
        end
        pn[:, it] .= x[Dim * npl .+ (1:npl)]
        Ln[:, :, it] .= hdg_recover_gradient(view(mesh.dgnodes, :, :, it), master,
                                             view(un, :, :, it), λe)
    end

    return (u=un, gradu=Ln, p=pn, Λ=Λn, ρ=ρ)
end

"""
    hdg_ns_solve(master, mesh, ν, dbc; τ=1.0, source=nothing, maxiter=20,
                 tol=1e-10, verbose=true, u0=nothing, Λ0=nothing)

Solves the steady incompressible Navier-Stokes equations (2D or 3D) with the
HDG method by Newton iteration (the first iteration, started from rest, is a
Stokes solve). See [`hdg_ns_step`](@ref) for the arguments and the returned
fields.
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

function hdg_cd_elemmat(dg, master, κ, τθ, ue, λe, θolde, dtinv, srce, sfun)
    Dim = ndims(master)
    nfe = Dim + 1
    nps = size(master.perm, 1)
    npl = size(dg, 1)
    nfc = nfe * nps
    shf = @view master.face.shap[:, 1, :]

    vol = hdg_elem_volume(dg, master)
    (; shap, shapd, M, C, wjac) = vol

    ug = shap' * ue                                     # (ng, Dim)
    Ku = @view(shapd[:, :, 1]) * Diagonal(@view(ug[:, 1])) * shap'
    for d in 2:Dim
        Ku .+= @view(shapd[:, :, d]) * Diagonal(@view(ug[:, d])) * shap'
    end

    E = zeros(npl, nfc, Dim)
    FN = zeros(npl, npl, Dim)
    Fτ = zeros(npl, npl)
    Bλ = zeros(npl, nfc)     # ⟨(λ·n) θ̂ - τθ θ̂, φ⟩
    HN = zeros(nfc, npl, Dim)
    Hτ = zeros(nfc, npl)
    Hλf = zeros(nfc, nfc)    # ⟨(λ·n) θ̂ - τθ θ̂, μ⟩

    for s in 1:nfe
        fc = hdg_elem_face(dg, master, s)
        (; ps, n, wds) = fc
        cols = (s - 1) * nps .+ (1:nps)

        λg = shf' * λe[cols, :]
        λn = @view(λg[:, 1]) .* @view(n[:, 1])
        for d in 2:Dim
            λn = λn .+ @view(λg[:, d]) .* @view(n[:, d])
        end

        T0 = facemat(shf, wds)
        Tλn = facemat(shf, wds .* λn)

        for d in 1:Dim
            Tn = facemat(shf, wds .* @view(n[:, d]))
            E[ps, cols, d] .+= Tn
            FN[ps, ps, d] .+= Tn
            HN[cols, ps, d] .+= Tn
        end
        Fτ[ps, ps] .+= τθ .* T0
        Bλ[ps, cols] .+= Tλn .- τθ .* T0
        Hτ[cols, ps] .+= τθ .* T0
        Hλf[cols, cols] .+= Tλn .- τθ .* T0
    end

    # Eliminate the gradient: q_d = M⁻¹ (E_d θ̂ - C_dᵀ θ)
    MF = cholesky(Symmetric(M))
    A = dtinv .* M .- Ku .+ Fτ
    B = copy(Bλ)
    Hx = copy(Hτ)                                       # flux carries -κ q·n
    Hλ = copy(Hλf)
    for d in 1:Dim
        MiCd = MF \ Matrix(@view(C[:, :, d])')
        MiEd = MF \ @view(E[:, :, d])
        Gd = κ .* (@view(C[:, :, d])' .- @view(FN[:, :, d]))
        A .-= Gd * MiCd
        B .+= Gd * MiEd
        Hx .+= κ .* (@view(HN[:, :, d]) * MiCd)
        Hλ .-= κ .* (@view(HN[:, :, d]) * MiEd)
    end

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

    Z = lu!(A) \ hcat(B, r)
    HxZ = Hx * Z
    Ke = Hλ .- HxZ[:, 1:nfc]
    re = -HxZ[:, nfc + 1]

    return Ke, re, Z
end

"Recover the scalar gradient q_d = M⁻¹(E_d θ̂ - C_dᵀ θ) of one element."
function hdg_recover_scalargrad(dg, master, θe, θ̂e)
    Dim = ndims(master)
    vol = hdg_elem_volume(dg, master)
    E = hdg_face_ops(dg, master)
    MF = cholesky(Symmetric(vol.M))
    q = zeros(size(dg, 1), Dim)
    for d in 1:Dim
        q[:, d] .= MF \ (@view(E[:, :, d]) * θ̂e .- @view(vol.C[:, :, d])' * θe)
    end
    return q
end

"""
    hdg_cd_step(master, mesh, κ, tbc; τ=1.0, u=nothing, Λ=nothing, θold=nothing,
                dtinv=0.0, source=nothing)

Solves one (linear) implicit step of the scalar HDG advection-diffusion
equation ∂θ/∂t + ∇·(uθ) - κΔθ = s with the DG velocity field `u`
(npl × Dim × nt) and the velocity trace `Λ` from the HDG Navier-Stokes solver.

`tbc(p, tag)` prescribes the boundary condition on a boundary face with tag
`tag` (`-mesh.f[i, end]`) at node coordinates `p`: return `(:d, value)` for a
Dirichlet condition on θ̂ or `(:n, flux)` for a prescribed total normal flux
(e.g. `(:n, 0.0)` for an insulated wall).

Returns a named tuple `(θ, q, Θ)` with the scalar field (npl × nt), its
gradient q = ∇θ (npl × Dim × nt), and the face trace.
"""
function hdg_cd_step(master, mesh, κ, tbc; τ=1.0, u=nothing, Λ=nothing, θold=nothing,
                     dtinv=0.0, source=nothing)
    Dim = ndims(master)
    nfe = Dim + 1
    nps = size(master.perm, 1)
    npl = size(mesh.dgnodes, 1)
    nt = size(mesh.t, 1)
    nf = size(mesh.f, 1)
    nfc = nfe * nps
    ndof = nps * nf
    elcon = mesh.elcon

    u === nothing && (u = zeros(npl, Dim, nt))
    Λ === nothing && (Λ = zeros(Dim * nps * nf))

    src = source isa AbstractArray ? source : nothing
    sfun = source isa Function ? source : nothing

    Kes = zeros(nfc, nfc, nt)
    res = zeros(nfc, nt)
    Zs = zeros(npl, nfc + 1, nt)

    Threads.@threads for it in 1:nt
        λe = gather_face_vector(Λ, elcon, it, nps, Dim)
        θolde = θold === nothing ? nothing : view(θold, :, it)
        srce = src === nothing ? nothing : view(src, :, it)
        Ke, re, Z = hdg_cd_elemmat(view(mesh.dgnodes, :, :, it), master, κ, τ,
                                   view(u, :, :, it), λe, θolde, dtinv, srce, sfun)
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
    shf = @view master.face.shap[:, 1, :]
    for i in 1:nf
        mesh.f[i, end] >= 0 && continue
        tag = -mesh.f[i, end]
        Xq, wds, Tm = boundary_face_quad(mesh, master, i)
        bc = [tbc(view(Xq, g, :), tag) for g in axes(Xq, 1)]
        vals = Float64[b[2] for b in bc]
        if bc[1][1] == :d
            gproj = Tm \ (shf * (wds .* vals))
            for k in 1:nps
                gd = (i - 1) * nps + k
                isdbc[gd] = true
                gvals[gd] = gproj[k]
            end
        else
            nrhs[(i - 1) * nps .+ (1:nps)] .+= shf * (wds .* vals)
        end
    end

    II = Int[]
    JJ = Int[]
    VV = Float64[]
    rhs = zeros(ndof)
    gdof = zeros(Int, nfc)

    for it in 1:nt
        for s in 1:nfe, a in 1:nps
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
    qn = zeros(npl, Dim, nt)
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

Two-dimensional only: the 3D variant needs the vector vorticity and a basis of
the face-tangential space and is not implemented yet.

`master1`/`mesh1` hold the degree k+1 approximation on the same triangulation
and must be built with the same quadrature order as `master`/`mesh` (see
[`hdg_postprocess`](@ref) for the same convention; on curved meshes `mesh1`
must also carry the same discrete geometry — see [`match_geometry!`](@ref)).
Returns u* as a (npl1 × 2 × nt) array on the nodes of `mesh1`.
"""
function hdg_ns_postprocess(master, mesh, master1, mesh1, result)
    ndims(master) == 2 ||
        throw(ArgumentError("hdg_ns_postprocess implements the 2D divergence-free postprocessing (paper §3.3); the 3D H(div) variant is not implemented"))
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

    sh1d0 = master.face.shap[:, 1, :]
    sh1d1 = master1.face.shap[:, 1, :]
    sh1d1x = master1.face.shap[:, 2, :]
    gw1d = master.face.gwgh
    shap1 = master1.shap[:, 1, :]

    # the single P_{k+1}(F)^⊥ test function: orthogonalize the top Koornwinder
    # mode against P_k(F) in the reference inner product (exact for straight
    # edges; its tangential derivative is evaluated through the edge metric)
    f1d, fx1d = koornwinder1d(vec(master.face.gpts), porder + 1)

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
        λe = gather_face_vector(Λ, elcon, it, nps, 2)

        E = zeros(nrow, 2npl1)
        b = zeros(nrow)
        cu1 = 1:npl1
        cu2 = npl1 .+ (1:npl1)

        for s in 1:3
            ed = hdg_elem_face(dg1, master1, s)
            n1, n2 = @view(ed.n[:, 1]), @view(ed.n[:, 2])
            (; ds, wds) = ed
            ip1 = master1.perm[:, s, 1]
            cols = (s - 1) * nps .+ (1:nps)

            # ⟨(u* - û)·n, μ⟩_F = 0, μ ∈ P_k(F)
            rows = ra .+ cols
            E[rows, cu1[ip1]] .+= sh1d0 * Diagonal(wds .* n1) * sh1d1'
            E[rows, cu2[ip1]] .+= sh1d0 * Diagonal(wds .* n2) * sh1d1'
            λn = (sh1d0' * λe[cols, 1]) .* n1 .+ (sh1d0' * λe[cols, 2]) .* n2
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
            i_f = mesh.t2f[it, s]
            jt = mesh.f[i_f, 3] == it ? mesh.f[i_f, 4] : mesh.f[i_f, 3]
            if jt > 0
                s2 = findfirst(==(i_f), mesh.t2f[jt, :])
                a = (a .+ nLt_edge(L, jt, s2, master, n1, n2; rev=true)) ./ 2
            end
            b[row] = sum(gw1d .* μx .* a)
        end

        # (u* - u, ∇w)_K = 0, w ∈ P_k(K)
        rows = rc .+ (1:npl)
        E[rows, cu1] .= @view(vol0.shapd[:, :, 1]) * shap1'
        E[rows, cu2] .= @view(vol0.shapd[:, :, 2]) * shap1'
        b[rows] .= @view(vol0.shapd[:, :, 1]) * (vol0.shap' * u[:, 1, it]) .+
                   @view(vol0.shapd[:, :, 2]) * (vol0.shap' * u[:, 2, it])

        # (∇×u* - w_h, w b_K)_K = 0, w ∈ P_{k-1}(K)
        Ψ = Wd .* bk
        rows = rd .+ (1:nk1)
        E[rows, cu1] .= -Ψ' * @view(vol1.shapd[:, :, 2])'
        E[rows, cu2] .= Ψ' * @view(vol1.shapd[:, :, 1])'
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
# match the orientation of the neighboring element across the face (2D only)
function nLt_edge(L, it, s, master, n1, n2; rev=false)
    sh1d0 = @view master.face.shap[:, 1, :]
    ps = master.perm[:, s, 1]
    t1, t2 = -n2, n1
    Lg = ntuple(c -> sh1d0' * (rev ? reverse(L[ps, c, it]) : L[ps, c, it]), 4)
    return n1 .* (Lg[1] .* t1 .+ Lg[2] .* t2) .+ n2 .* (Lg[3] .* t1 .+ Lg[4] .* t2)
end
