# Batched HDG Navier-Stokes and scalar-transport assembly on any KA backend
# (Phase 5 of GPU_PLAN.md). The per-element math of `hdg_ns_elemmat` /
# `hdg_cd_elemmat` splits into
#
#   - geometry × (ν, τ) constants (mass, coupling, edge lifts and the
#     eliminated-gradient viscous composites Avisc/Bvisc/Hvisc/Hλvisc),
#     built once on the CPU with the same helpers the legacy path uses; and
#   - the Newton-/state-varying convection linearization (volume terms from
#     `u` at the volume quadrature points, trace terms from λ at the face
#     quadrature points), rebuilt each call as KA kernels + one batched
#     in-kernel LU (`_blusolve!`) — no per-element factorizations.
#
# The condensed trace saddle-point system stays a CPU sparse LU, but its
# sparsity pattern (`II/JJ`, connectivity-only) is precomputed once and the
# UMFPACK factorization is reused numerically (`lu!`) across Newton
# iterations and time steps. Recovery of (u, p, L) — and (θ, q) for the
# scalar step — is batched matvecs on the device.
#
# Everything runs through KernelAbstractions + Adapt (`ArrayT = CuArray`,
# `ROCArray`, … — no vendor-specific code); `hdg_ns_step` / `hdg_cd_step`
# stay untouched as the parity reference.

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using Adapt
using LinearAlgebra
using SparseArrays
using SparseArrays: UMFPACK

# UMFPACK on the NS trace saddle-point system: the zero-diagonal ρ block makes
# the default pivoting explode with fill (≈8× slower at 8k elements). The
# symmetric strategy + relaxed pivot tolerance keeps the fill ordering-driven;
# UMFPACK's built-in iterative refinement holds the solve at ~1e-9 relative.
# Guarded so any UMFPACK API drift degrades to the default factorization.
function _ns_trace_lu(H)
    ctrl = try
        c = UMFPACK.get_umfpack_control(Float64, Int64)
        c[6] = 3.0                                    # UMFPACK_STRATEGY_SYMMETRIC
        c[UMFPACK.JL_UMFPACK_PIVOT_TOLERANCE] = 1e-4
        c[UMFPACK.JL_UMFPACK_SYM_PIVOT_TOLERANCE] = 1e-4
        c
    catch
        nothing
    end
    return ctrl === nothing ? lu(H) : lu(H; control=ctrl)
end

# ------------------------------------------------------------------------------
# Navier-Stokes batch: per-element constants
# ------------------------------------------------------------------------------

"""
    HDGNSBatch(master, mesh, ν, τ; T=Float64)

Per-element constant operator matrices of the batched HDG Navier-Stokes
Newton step, built once on the CPU (all fields plain arrays,
`Adapt.@adapt_structure`). With `npl` volume nodes, `nps = porder + 1` face
nodes per edge, `nfc = 3nps`, the local solve size `nv = 3npl` (u1, u2, p)
and `ncB = 2nfc + 2` right-hand-side columns (trace columns + bρ + r):

- `shap (npl, ng)`, `sh1d (nps, nq1d)`: shared shape values at quadrature.
- `shapx/shapy (npl, ng, nt)`: weighted physical derivative matrices.
- `M (npl, npl, nt)`: element mass (the `dtinv·M` term is added per call).
- `A0 (nv, nv, nt)`: constant part of the local Newton matrix, including the
  viscous composite and the mean-pressure gauge row.
- `Bhat0 (nv, ncB, nt)`: constant part of the local RHS block `[B | bρ | r]`
  (`r` column zero).
- `Hx (2nfc, nv, nt)`, `Hlam0 (2nfc, 2nfc, nt)`: flux-continuity test blocks
  (fully constant, resp. constant part).
- `MiE1/MiE2 (npl, nfc, nt)`, `MiCx/MiCy (npl, npl, nt)`: gradient-recovery
  maps `M⁻¹E_j`, `M⁻¹C_jᵀ`.
- `wds/fn1/fn2 (nq1d, 3, nt)`: face quadrature measure and unit normal.
- `perm (nps, 3)`, `elcon (nps, 3, nt)` (`Int32`): edge-to-volume node map
  and global face-node connectivity.
- `crow (2nfc, nt)`, `area (nt)`: compatibility row and element area
  (consumed on the host during global assembly).
"""
struct HDGNSBatch{T, A1 <: AbstractVector{T}, A2 <: AbstractMatrix{T},
                  A3 <: AbstractArray{T, 3}, I2 <: AbstractMatrix{Int32},
                  I3 <: AbstractArray{Int32, 3}}
    npl   :: Int
    nps   :: Int
    nfc   :: Int
    nt    :: Int
    ng    :: Int
    nq1d  :: Int
    shap  :: A2
    sh1d  :: A2
    shapx :: A3
    shapy :: A3
    M     :: A3
    A0    :: A3
    Bhat0 :: A3
    Hx    :: A3
    Hlam0 :: A3
    MiE1  :: A3
    MiE2  :: A3
    MiCx  :: A3
    MiCy  :: A3
    wds   :: A3
    fn1   :: A3
    fn2   :: A3
    crow  :: A2
    area  :: A1
    perm  :: I2
    elcon :: I3
end

Adapt.@adapt_structure HDGNSBatch

Base.eltype(::HDGNSBatch{T}) where {T} = T

function HDGNSBatch(master, mesh, ν, τ; T::Type{<:AbstractFloat}=Float64)
    nps = master.porder + 1
    nfc = 3 * nps
    npl = size(mesh.dgnodes, 1)
    nt = size(mesh.t, 1)
    ng = length(master.gwgh)
    nq1d = length(master.gw1d)
    nv = 3 * npl
    ncB = 2 * nfc + 2

    shap2 = Matrix(master.shap[:, 1, :])
    sh1d = Matrix(master.sh1d[:, 1, :])

    shapx = zeros(npl, ng, nt)
    shapy = zeros(npl, ng, nt)
    M = zeros(npl, npl, nt)
    A0 = zeros(nv, nv, nt)
    Bhat0 = zeros(nv, ncB, nt)
    Hx = zeros(2 * nfc, nv, nt)
    Hlam0 = zeros(2 * nfc, 2 * nfc, nt)
    MiE1 = zeros(npl, nfc, nt)
    MiE2 = zeros(npl, nfc, nt)
    MiCx = zeros(npl, npl, nt)
    MiCy = zeros(npl, npl, nt)
    wdsb = zeros(nq1d, 3, nt)
    fn1 = zeros(nq1d, 3, nt)
    fn2 = zeros(nq1d, 3, nt)
    crow = zeros(2 * nfc, nt)
    area = zeros(nt)

    iu1, iu2, ip = 1:npl, npl .+ (1:npl), 2npl .+ (1:npl)
    j1, j2 = 1:nfc, nfc .+ (1:nfc)
    icp = 2npl + 1

    @views Threads.@threads for it in 1:nt
        dg = mesh.dgnodes[:, :, it]
        vol = hdg_elem_volume(dg, master)
        shapx[:, :, it] .= vol.shapx
        shapy[:, :, it] .= vol.shapy
        M[:, :, it] .= vol.M
        area[it] = sum(vol.wjac)

        E1 = zeros(npl, nfc)
        E2 = zeros(npl, nfc)
        FN1 = zeros(npl, npl)
        FN2 = zeros(npl, npl)
        Fτ = zeros(npl, npl)
        Eτ = zeros(npl, nfc)
        HN1 = zeros(nfc, npl)
        HN2 = zeros(nfc, npl)
        Hτ = zeros(nfc, npl)
        Hλτ = zeros(nfc, nfc)

        for s in 1:3
            ed = hdg_elem_edge(dg, master, s)
            cols = (s - 1) * nps .+ (1:nps)
            T0 = facemat(sh1d, ed.wds)
            Tn1 = facemat(sh1d, ed.wds .* ed.n1)
            Tn2 = facemat(sh1d, ed.wds .* ed.n2)
            E1[ed.ps, cols] .+= Tn1
            E2[ed.ps, cols] .+= Tn2
            FN1[ed.ps, ed.ps] .+= Tn1
            FN2[ed.ps, ed.ps] .+= Tn2
            Fτ[ed.ps, ed.ps] .+= τ .* T0
            Eτ[ed.ps, cols] .+= τ .* T0
            HN1[cols, ed.ps] .+= Tn1
            HN2[cols, ed.ps] .+= Tn2
            Hτ[cols, ed.ps] .+= τ .* T0
            Hλτ[cols, cols] .+= τ .* T0
            crow[cols, it] .+= sh1d * (ed.wds .* ed.n1)
            crow[nfc .+ cols, it] .+= sh1d * (ed.wds .* ed.n2)
            wdsb[:, s, it] .= ed.wds
            fn1[:, s, it] .= ed.n1
            fn2[:, s, it] .= ed.n2
        end

        MF = cholesky(Symmetric(Matrix(vol.M)))
        MiCx_ = MF \ Matrix(vol.Cx')
        MiCy_ = MF \ Matrix(vol.Cy')
        MiE1_ = MF \ E1
        MiE2_ = MF \ E2
        MiE1[:, :, it] .= MiE1_
        MiE2[:, :, it] .= MiE2_
        MiCx[:, :, it] .= MiCx_
        MiCy[:, :, it] .= MiCy_

        G1 = ν .* (vol.Cx' .- FN1)
        G2 = ν .* (vol.Cy' .- FN2)
        Avisc = .-(G1 * MiCx_ .+ G2 * MiCy_)
        Bvisc = G1 * MiE1_ .+ G2 * MiE2_
        Hvisc = ν .* (HN1 * MiCx_ .+ HN2 * MiCy_)
        Hλvisc = .-ν .* (HN1 * MiE1_ .+ HN2 * MiE2_)

        A0[iu1, iu1, it] .= Avisc .+ Fτ
        A0[iu2, iu2, it] .= Avisc .+ Fτ
        A0[iu1, ip, it] .= .-vol.Cx' .+ FN1
        A0[iu2, ip, it] .= .-vol.Cy' .+ FN2
        A0[ip, iu1, it] .= .-vol.Cx'
        A0[ip, iu2, it] .= .-vol.Cy'
        A0[icp, :, it] .= 0.0
        A0[icp, ip, it] .= shap2 * vol.wjac

        Bhat0[iu1, j1, it] .= Bvisc .- Eτ
        Bhat0[iu2, j2, it] .= Bvisc .- Eτ
        Bhat0[ip, j1, it] .= E1
        Bhat0[ip, j2, it] .= E2
        Bhat0[icp, :, it] .= 0.0
        Bhat0[icp, 2 * nfc + 1, it] = -area[it]

        Hx[j1, iu1, it] .= Hvisc .+ Hτ
        Hx[j2, iu2, it] .= Hvisc .+ Hτ
        Hx[j1, ip, it] .= HN1
        Hx[j2, ip, it] .= HN2

        Hlam0[j1, j1, it] .= Hλvisc .- Hλτ
        Hlam0[j2, j2, it] .= Hλvisc .- Hλτ
    end

    perm = Int32.(master.perm[:, :, 1])
    elcon = Int32.(mesh.elcon)

    return HDGNSBatch(npl, nps, nfc, nt, ng, nq1d,
                      T.(shap2), T.(sh1d), T.(shapx), T.(shapy), T.(M),
                      T.(A0), T.(Bhat0), T.(Hx), T.(Hlam0),
                      T.(MiE1), T.(MiE2), T.(MiCx), T.(MiCy),
                      T.(wdsb), T.(fn1), T.(fn2), T.(crow), T.(area),
                      perm, elcon)
end

# ------------------------------------------------------------------------------
# KA kernels (state-varying assembly + recovery)
# ------------------------------------------------------------------------------

# quad-point velocity values ug[g, c, e] = Σ_m shap[m, g] u[m, c, e]
@kernel function _ns_quadvel!(ug, @Const(shap), @Const(u))
    g, c, e = @index(Global, NTuple)
    T = eltype(ug)
    npl = size(u, 1)
    acc = zero(T)
    @inbounds for m in 1:npl
        acc += shap[m, g] * u[m, c, e]
    end
    @inbounds ug[g, c, e] = acc
end

# X[m, n, e] = Σ_g A[m, g, e] * ug[g, ci, e] * shap[n, g]  (A = shapx or shapy)
@kernel function _ns_wgemm!(X, @Const(A), @Const(ug), ci, @Const(shap))
    m, n, e = @index(Global, NTuple)
    T = eltype(X)
    ng = size(A, 2)
    acc = zero(T)
    @inbounds for g in 1:ng
        acc += A[m, g, e] * ug[g, ci, e] * shap[n, g]
    end
    @inbounds X[m, n, e] = acc
end

# A = A0 + Newton convection blocks + dtinv·M on the velocity diagonal blocks.
# Block (1,1) gets -X1 - Ku = -2X1 - Y2, block (2,2) gets -Y2 - Ku = -X1 - 2Y2,
# off-diagonal velocity blocks -Y1 / -X2; pressure rows/cols (incl. the gauge
# row) are pure A0.
@kernel function _ns_assemble_A!(A, @Const(A0), @Const(X1), @Const(X2),
                                 @Const(Y1), @Const(Y2), @Const(M), dtinv, npl)
    i, j, e = @index(Global, NTuple)
    @inbounds begin
        base = A0[i, j, e]
        if i <= npl && j <= npl
            base += -2 * X1[i, j, e] - Y2[i, j, e] + dtinv * M[i, j, e]
        elseif i <= npl && j <= 2 * npl
            base += -Y1[i, j - npl, e]
        elseif i <= 2 * npl && j <= npl
            base += -X2[i - npl, j, e]
        elseif i <= 2 * npl && j <= 2 * npl
            base += -X1[i - npl, j - npl, e] - 2 * Y2[i - npl, j - npl, e] +
                    dtinv * M[i - npl, j - npl, e]
        end
        A[i, j, e] = base
    end
end

# r column of Bhat: r_c += -(Ku u)_c + M (fsrc_c + dtinv uold_c) + rext_c,
# using (Ku u)_c[i] = Σ_g (shapx[i,g] ug1 + shapy[i,g] ug2) ug_c.
@kernel function _ns_rhs!(Bhat, @Const(shapx), @Const(shapy), @Const(ug),
                          @Const(M), @Const(fsrc), @Const(uold), @Const(rext),
                          dtinv, npl, ncB)
    i, c, e = @index(Global, NTuple)
    T = eltype(Bhat)
    ng = size(ug, 1)
    acc = zero(T)
    @inbounds for g in 1:ng
        acc -= (shapx[i, g, e] * ug[g, 1, e] + shapy[i, g, e] * ug[g, 2, e]) * ug[g, c, e]
    end
    @inbounds for j in 1:npl
        acc += M[i, j, e] * (fsrc[j, c, e] + dtinv * uold[j, c, e])
    end
    @inbounds acc += rext[i, c, e]
    @inbounds Bhat[(c - 1) * npl + i, ncB, e] += acc
end

# Newton linearization of the trace convection (λ·n)λ_i, accumulated into the
# local RHS block Bhat, the trace-test block Hlam and the flux RHS rH. One
# workitem per element (the serial 3-edge loop sidesteps corner-node races).
@kernel function _ns_faces!(Bhat, Hlam, rH, @Const(lam), @Const(sh1d),
                            @Const(wds), @Const(fn1), @Const(fn2),
                            @Const(perm), npl, nps, ncB)
    e = @index(Global)
    T = eltype(Bhat)
    nfc = 3 * nps
    nq = size(sh1d, 2)
    @inbounds for s in 1:3
        for g in 1:nq
            λ1 = zero(T)
            λ2 = zero(T)
            for a in 1:nps
                col = (s - 1) * nps + a
                sh = sh1d[a, g]
                λ1 += sh * lam[col, e]
                λ2 += sh * lam[nfc + col, e]
            end
            n1 = fn1[g, s, e]
            n2 = fn2[g, s, e]
            w = wds[g, s, e]
            λn = λ1 * n1 + λ2 * n2
            for a in 1:nps
                sha = sh1d[a, g]
                p = perm[a, s]
                cola = (s - 1) * nps + a
                rc = w * sha * λn
                Bhat[p, ncB, e] += rc * λ1
                Bhat[npl + p, ncB, e] += rc * λ2
                rH[cola, e] += rc * λ1
                rH[nfc + cola, e] += rc * λ2
                for b in 1:nps
                    colb = (s - 1) * nps + b
                    wab = w * sha * sh1d[b, g]
                    c11 = wab * (n1 * λ1 + λn)
                    c12 = wab * n2 * λ1
                    c21 = wab * n1 * λ2
                    c22 = wab * (n2 * λ2 + λn)
                    Bhat[p, colb, e] += c11
                    Bhat[p, nfc + colb, e] += c12
                    Bhat[npl + p, colb, e] += c21
                    Bhat[npl + p, nfc + colb, e] += c22
                    Hlam[cola, colb, e] += c11
                    Hlam[cola, nfc + colb, e] += c12
                    Hlam[nfc + cola, colb, e] += c21
                    Hlam[nfc + cola, nfc + colb, e] += c22
                end
            end
        end
    end
end

# gather the interleaved global trace into element-local [λ1; λ2] blocks
@kernel function _ns_gather!(lam, @Const(Λ), @Const(elcon), nps)
    k, e = @index(Global, NTuple)
    nfc = 3 * nps
    c = k <= nfc ? 1 : 2
    ℓ = k - (c - 1) * nfc
    s = div(ℓ - 1, nps) + 1
    a = ℓ - (s - 1) * nps
    @inbounds g = elcon[a, s, e]
    @inbounds lam[k, e] = Λ[2 * (g - 1) + c]
end

# (u, p) recovery x = Z r-col − Z_B λ − Z_bρ ρ, routed into un/pn
@kernel function _ns_recover!(un, pn, @Const(Z), @Const(lam), @Const(ρ), npl, ncB)
    i, e = @index(Global, NTuple)
    T = eltype(un)
    nlam = ncB - 2
    @inbounds acc = Z[i, ncB, e] - Z[i, ncB - 1, e] * ρ[e]
    @inbounds for k in 1:nlam
        acc -= Z[i, k, e] * lam[k, e]
    end
    @inbounds if i <= npl
        un[i, 1, e] = acc
    elseif i <= 2 * npl
        un[i - npl, 2, e] = acc
    else
        pn[i - 2 * npl, e] = acc
    end
end

# velocity gradient L_ij = M⁻¹(E_j λ_i − C_jᵀ u_i), columns L11 L12 L21 L22
@kernel function _ns_gradient!(Ln, @Const(MiE1), @Const(MiE2), @Const(MiCx),
                               @Const(MiCy), @Const(lam), @Const(un), nfc)
    i, e = @index(Global, NTuple)
    T = eltype(Ln)
    npl = size(un, 1)
    l11 = zero(T); l12 = zero(T); l21 = zero(T); l22 = zero(T)
    @inbounds for k in 1:nfc
        λ1k = lam[k, e]
        λ2k = lam[nfc + k, e]
        l11 += MiE1[i, k, e] * λ1k
        l12 += MiE2[i, k, e] * λ1k
        l21 += MiE1[i, k, e] * λ2k
        l22 += MiE2[i, k, e] * λ2k
    end
    @inbounds for j in 1:npl
        u1j = un[j, 1, e]
        u2j = un[j, 2, e]
        l11 -= MiCx[i, j, e] * u1j
        l12 -= MiCy[i, j, e] * u1j
        l21 -= MiCx[i, j, e] * u2j
        l22 -= MiCy[i, j, e] * u2j
    end
    @inbounds begin
        Ln[i, 1, e] = l11
        Ln[i, 2, e] = l12
        Ln[i, 3, e] = l21
        Ln[i, 4, e] = l22
    end
end

# scalar-trace gather (face-scalar DOFs, no interleaving)
@kernel function _cd_gather!(th, @Const(Θ), @Const(elcon), nps)
    k, e = @index(Global, NTuple)
    s = div(k - 1, nps) + 1
    a = k - (s - 1) * nps
    @inbounds th[k, e] = Θ[elcon[a, s, e]]
end

# trace convection ⟨(λ·n) θ̂, ·⟩ into the local RHS block B and trace block Hlam
@kernel function _cd_faces!(B, Hlam, @Const(lam), @Const(sh1d), @Const(wds),
                            @Const(fn1), @Const(fn2), @Const(perm), nps, ncB)
    e = @index(Global)
    T = eltype(B)
    nfc = 3 * nps
    nq = size(sh1d, 2)
    @inbounds for s in 1:3
        for g in 1:nq
            λ1 = zero(T)
            λ2 = zero(T)
            for a in 1:nps
                col = (s - 1) * nps + a
                sh = sh1d[a, g]
                λ1 += sh * lam[col, e]
                λ2 += sh * lam[nfc + col, e]
            end
            λn = λ1 * fn1[g, s, e] + λ2 * fn2[g, s, e]
            w = wds[g, s, e]
            for a in 1:nps
                sha = sh1d[a, g]
                p = perm[a, s]
                cola = (s - 1) * nps + a
                for b in 1:nps
                    colb = (s - 1) * nps + b
                    c = w * sha * sh1d[b, g] * λn
                    B[p, colb, e] += c
                    Hlam[cola, colb, e] += c
                end
            end
        end
    end
end

# scalar RHS column: r += M (dtinv θold + fsrc) + rext
@kernel function _cd_rhs!(B, @Const(M), @Const(θold), @Const(fsrc),
                          @Const(rext), dtinv, ncB)
    i, e = @index(Global, NTuple)
    T = eltype(B)
    npl = size(M, 1)
    acc = zero(T)
    @inbounds for j in 1:npl
        acc += M[i, j, e] * (dtinv * θold[j, e] + fsrc[j, e])
    end
    @inbounds acc += rext[i, e]
    @inbounds B[i, ncB, e] += acc
end

# scalar recovery θ = Z r-col − Z_B θ̂
@kernel function _cd_recover!(θn, @Const(Z), @Const(th), ncB)
    i, e = @index(Global, NTuple)
    @inbounds acc = Z[i, ncB, e]
    @inbounds for k in 1:ncB-1
        acc -= Z[i, k, e] * th[k, e]
    end
    @inbounds θn[i, e] = acc
end

# scalar gradient q_j = M⁻¹(E_j θ̂ − C_jᵀ θ)
@kernel function _cd_gradq!(qn, @Const(MiE1), @Const(MiE2), @Const(MiCx),
                            @Const(MiCy), @Const(th), @Const(θn))
    i, e = @index(Global, NTuple)
    T = eltype(qn)
    nfc = size(MiE1, 2)
    npl = size(θn, 1)
    q1 = zero(T); q2 = zero(T)
    @inbounds for k in 1:nfc
        thk = th[k, e]
        q1 += MiE1[i, k, e] * thk
        q2 += MiE2[i, k, e] * thk
    end
    @inbounds for j in 1:npl
        θj = θn[j, e]
        q1 -= MiCx[i, j, e] * θj
        q2 -= MiCy[i, j, e] * θj
    end
    @inbounds qn[i, 1, e] = q1
    @inbounds qn[i, 2, e] = q2
end

# ------------------------------------------------------------------------------
# Trace-system pattern (host): connectivity-only, precomputed once
# ------------------------------------------------------------------------------

# NS trace saddle-point pattern, mirroring the hdg_ns_step assembly order
# exactly so the CSC structure (and the reusable numeric factorization) is
# identical every call — with one deliberate deviation: the zero-mean pressure
# gauge Σ|K|ρ_K = 0 is a *dense* row (all nt mean-pressure columns), which
# causes catastrophic UMFPACK fill-in (~10× the whole step at 8k elements).
# We pin ρ_1 = 0 instead (single diagonal entry) and restore the legacy gauge
# after the solve by the constant shift ρ .-= Σ|K|ρ_K / Σ|K|, p .-= shift —
# a constant pressure shift leaves u and λ exactly unchanged.
struct HDGNSTracePattern
    II    :: Vector{Int}
    JJ    :: Vector{Int}
    gdofs :: Matrix{Int}     # (2nfc, nt)
    isdbc :: BitVector
    ndof  :: Int
    nΛ    :: Int
end

function HDGNSTracePattern(mesh)
    nps = mesh.porder + 1
    nfc = 3 * nps
    nt = size(mesh.t, 1)
    nf = size(mesh.f, 1)
    nΛ = 2 * nps * nf
    ndof = nΛ + nt
    elcon = mesh.elcon

    isdbc = falses(ndof)
    for i in 1:nf
        mesh.f[i, 4] >= 0 && continue
        for k in 1:nps, c in 1:2
            isdbc[2 * ((i - 1) * nps + k - 1) + c] = true
        end
    end

    gdofs = zeros(Int, 2 * nfc, nt)
    for it in 1:nt, s in 1:3, a in 1:nps
        ℓ = (s - 1) * nps + a
        gdofs[ℓ, it] = 2 * elcon[a, s, it] - 1
        gdofs[nfc + ℓ, it] = 2 * elcon[a, s, it]
    end

    II = Int[]
    JJ = Int[]
    sizehint!(II, nt * (2 * nfc + 2)^2)
    sizehint!(JJ, nt * (2 * nfc + 2)^2)
    for it in 1:nt
        for jl in 1:2*nfc
            gj = gdofs[jl, it]
            for il in 1:2*nfc
                gi = gdofs[il, it]
                isdbc[gi] && continue
                push!(II, gi); push!(JJ, gj)
            end
            if it != 1
                push!(II, nΛ + it); push!(JJ, gj)
            end
        end
        for il in 1:2*nfc
            gi = gdofs[il, it]
            isdbc[gi] && continue
            push!(II, gi); push!(JJ, nΛ + it)
        end
    end
    push!(II, nΛ + 1); push!(JJ, nΛ + 1)          # ρ_1 = 0 gauge (sparse)
    for gd in 1:nΛ
        if isdbc[gd]
            push!(II, gd); push!(JJ, gd)
        end
    end

    return HDGNSTracePattern(II, JJ, gdofs, isdbc, ndof, nΛ)
end

# Refill the numeric values / RHS in the exact pattern order.
function ns_trace_fill!(VV, rhs, pat::HDGNSTracePattern, Kes, Ges, res, crows,
                        gvals)
    nfc2 = size(Kes, 1)
    nt = size(Kes, 3)
    n = 0
    fill!(rhs, 0.0)
    @inbounds for it in 1:nt
        for jl in 1:nfc2
            for il in 1:nfc2
                gi = pat.gdofs[il, it]
                pat.isdbc[gi] && continue
                n += 1
                VV[n] = Kes[il, jl, it]
            end
            if it != 1
                n += 1
                VV[n] = crows[jl, it]
            end
        end
        for il in 1:nfc2
            gi = pat.gdofs[il, it]
            pat.isdbc[gi] && continue
            n += 1
            VV[n] = Ges[il, it]
            rhs[gi] += res[il, it]
        end
    end
    n += 1
    VV[n] = 1.0                                    # ρ_1 = 0 gauge row
    @inbounds for gd in 1:pat.nΛ
        if pat.isdbc[gd]
            n += 1
            VV[n] = 1.0
            rhs[gd] = gvals[gd]
        end
    end
    @assert n == length(VV)
    return VV, rhs
end

# ------------------------------------------------------------------------------
# NS cache + driver
# ------------------------------------------------------------------------------

mutable struct HDGNSCache{B <: HDGNSBatch, W}
    batch :: B
    pat   :: HDGNSTracePattern
    work  :: W
    crowh :: Matrix{Float64}
    areah :: Vector{Float64}
    pgh   :: Array{Float64, 3}    # (ng, 2, nt) quad coords for function sources
    wjach :: Matrix{Float64}      # (ng, nt)
    shaph :: Matrix{Float64}      # (npl, ng)
    VV    :: Vector{Float64}
    rhs   :: Vector{Float64}
    F     :: Any                  # UMFPACK factorization, reused via lu!
    ν     :: Float64
    τ     :: Float64
end

function _ns_make_work(backend, T, npl, nps, nfc, ng, nt, nΛ)
    nv = 3 * npl
    ncB = 2 * nfc + 2
    kz(dims...) = KernelAbstractions.zeros(backend, T, dims...)
    return (; ug = kz(ng, 2, nt),
            X1 = kz(npl, npl, nt), X2 = kz(npl, npl, nt),
            Y1 = kz(npl, npl, nt), Y2 = kz(npl, npl, nt),
            A = kz(nv, nv, nt), Bhat = kz(nv, ncB, nt),
            HxZ = kz(2 * nfc, ncB, nt), Hlam = kz(2 * nfc, 2 * nfc, nt),
            rH = kz(2 * nfc, nt), lam = kz(2 * nfc, nt),
            u_d = kz(npl, 2, nt), fsrc = kz(npl, 2, nt),
            uold_d = kz(npl, 2, nt), rext = kz(npl, 2, nt),
            Λ_d = kz(nΛ), ρ_d = kz(nt),
            un = kz(npl, 2, nt), pn = kz(npl, nt), Ln = kz(npl, 4, nt))
end

function HDGNSCache(master, mesh, ν, τ; ArrayT=Array, T::Type{<:AbstractFloat}=Float64)
    batch_h = HDGNSBatch(master, mesh, ν, τ; T)
    crowh = Float64.(batch_h.crow)
    areah = Float64.(batch_h.area)

    npl, ng, nt = batch_h.npl, batch_h.ng, batch_h.nt
    pgh = zeros(ng, 2, nt)
    wjach = zeros(ng, nt)
    shaph = Matrix(master.shap[:, 1, :])
    @views Threads.@threads for it in 1:nt
        vol = hdg_elem_volume(mesh.dgnodes[:, :, it], master)
        pgh[:, :, it] .= shaph' * mesh.dgnodes[:, :, it]
        wjach[:, it] .= vol.wjac
    end

    batch = adapt(ArrayT, batch_h)
    pat = HDGNSTracePattern(mesh)
    backend = KernelAbstractions.get_backend(batch.M)
    nΛ = 2 * batch.nps * size(mesh.f, 1)
    work = _ns_make_work(backend, T, batch.npl, batch.nps, batch.nfc, batch.ng,
                         batch.nt, nΛ)
    return HDGNSCache(batch, pat, work, crowh, areah, pgh, wjach, shaph,
                      zeros(length(pat.II)), zeros(pat.ndof), nothing,
                      Float64(ν), Float64(τ))
end

# Dirichlet trace DOFs: λ = P_∂g, the face L2 projection of the data (same as
# hdg_ns_step).
function _ns_dirichlet_gvals!(gvals, mesh, master, dbc, nps)
    sh1d = @view master.sh1d[:, 1, :]
    for i in axes(mesh.f, 1)
        mesh.f[i, 4] >= 0 && continue
        Xq, wds, Tm = boundary_face_quad(mesh, master, i)
        gq = reduce(hcat, dbc(view(Xq, g, :)) for g in axes(Xq, 1))
        for c in 1:2
            gproj = Tm \ (sh1d * (wds .* gq[c, :]))
            for k in 1:nps
                gvals[2 * ((i - 1) * nps + k - 1) + c] = gproj[k]
            end
        end
    end
    return gvals
end

"""
    hdg_ns_step_batched(master, mesh, ν, dbc; τ=1.0, source=nothing, u=nothing,
                        Λ=nothing, uold=nothing, dtinv=0.0,
                        ArrayT=Array, T=Float64, cache=nothing)

Batched/KA counterpart of [`hdg_ns_step`](@ref) (same arguments, same returned
fields plus `cache`): the per-element Newton assembly, local solves and
(u, p, L) recovery run on the backend of `ArrayT` (e.g. `CuArray`); the
condensed trace saddle-point system is solved on the CPU with a sparse LU
whose sparsity pattern and factorization are reused across calls.

Pass the returned `cache` back in on subsequent calls (same `master`, `mesh`,
`ν`, `τ`) to skip all setup and reuse the factorization.
"""
function hdg_ns_step_batched(master, mesh, ν, dbc; τ=1.0, source=nothing,
                             u=nothing, Λ=nothing, uold=nothing, dtinv=0.0,
                             ArrayT=Array, T::Type{<:AbstractFloat}=Float64,
                             cache=nothing)
    if cache === nothing
        cache = HDGNSCache(master, mesh, ν, τ; ArrayT, T)
    else
        (cache.ν == ν && cache.τ == τ) ||
            throw(ArgumentError("cache was built with ν=$(cache.ν), τ=$(cache.τ)"))
    end
    batch, work, pat = cache.batch, cache.work, cache.pat
    backend = KernelAbstractions.get_backend(batch.M)
    npl, nps, nfc, ng, nt = batch.npl, batch.nps, batch.nfc, batch.ng, batch.nt
    nv = 3 * npl
    ncB = 2 * nfc + 2
    nΛ = pat.nΛ

    # state and sources to the device
    u === nothing ? fill!(work.u_d, zero(T)) : copyto!(work.u_d, T.(u))
    Λ === nothing ? fill!(work.Λ_d, zero(T)) : copyto!(work.Λ_d, T.(Λ))
    uold === nothing ? fill!(work.uold_d, zero(T)) : copyto!(work.uold_d, T.(uold))
    if source isa AbstractArray
        copyto!(work.fsrc, T.(source))
        fill!(work.rext, zero(T))
    elseif source isa Function
        fill!(work.fsrc, zero(T))
        rext = zeros(npl, 2, nt)
        @views Threads.@threads for it in 1:nt
            fg = reduce(hcat, source(cache.pgh[g, :, it]) for g in 1:ng)
            rext[:, 1, it] .= cache.shaph * (cache.wjach[:, it] .* fg[1, :])
            rext[:, 2, it] .= cache.shaph * (cache.wjach[:, it] .* fg[2, :])
        end
        copyto!(work.rext, T.(rext))
    else
        fill!(work.fsrc, zero(T))
        fill!(work.rext, zero(T))
    end

    # device: Newton-linearized local systems + static condensation
    _ns_quadvel!(backend)(work.ug, batch.shap, work.u_d; ndrange=(ng, 2, nt))
    _ns_wgemm!(backend)(work.X1, batch.shapx, work.ug, 1, batch.shap; ndrange=(npl, npl, nt))
    _ns_wgemm!(backend)(work.X2, batch.shapx, work.ug, 2, batch.shap; ndrange=(npl, npl, nt))
    _ns_wgemm!(backend)(work.Y1, batch.shapy, work.ug, 1, batch.shap; ndrange=(npl, npl, nt))
    _ns_wgemm!(backend)(work.Y2, batch.shapy, work.ug, 2, batch.shap; ndrange=(npl, npl, nt))
    _ns_assemble_A!(backend)(work.A, batch.A0, work.X1, work.X2, work.Y1,
                             work.Y2, batch.M, T(dtinv), npl; ndrange=(nv, nv, nt))
    copyto!(work.Bhat, batch.Bhat0)
    copyto!(work.Hlam, batch.Hlam0)
    fill!(work.rH, zero(T))
    _ns_gather!(backend)(work.lam, work.Λ_d, batch.elcon, nps; ndrange=(2 * nfc, nt))
    _ns_rhs!(backend)(work.Bhat, batch.shapx, batch.shapy, work.ug, batch.M,
                      work.fsrc, work.uold_d, work.rext, T(dtinv), npl, ncB;
                      ndrange=(npl, 2, nt))
    _ns_faces!(backend)(work.Bhat, work.Hlam, work.rH, work.lam, batch.sh1d,
                        batch.wds, batch.fn1, batch.fn2, batch.perm, npl, nps,
                        ncB; ndrange=nt)
    _blusolve!(backend)(work.A, work.Bhat; ndrange=nt)        # Bhat := Z
    _bgemm_nn!(backend)(work.HxZ, batch.Hx, work.Bhat, one(T), zero(T);
                        ndrange=(2 * nfc, ncB, nt))
    work.Hlam .-= view(work.HxZ, :, 1:2*nfc, :)               # Ke
    work.rH .-= view(work.HxZ, :, ncB, :)                     # re
    KernelAbstractions.synchronize(backend)

    Kes = Float64.(Array(work.Hlam))
    Ges = -Float64.(Array(work.HxZ[:, ncB - 1, :]))
    res = Float64.(Array(work.rH))

    # host: Dirichlet data, global assembly (precomputed pattern), sparse LU
    gvals = zeros(pat.ndof)
    _ns_dirichlet_gvals!(gvals, mesh, master, dbc, nps)
    ns_trace_fill!(cache.VV, cache.rhs, pat, Kes, Ges, res, cache.crowh, gvals)
    H = sparse(pat.II, pat.JJ, cache.VV, pat.ndof, pat.ndof)
    if cache.F === nothing
        cache.F = _ns_trace_lu(H)
    else
        lu!(cache.F, H)
    end
    sol = cache.F \ cache.rhs
    Λn = sol[1:nΛ]
    ρ = sol[nΛ+1:end]
    # restore the legacy zero-mean gauge Σ|K|ρ_K = 0 (constant pressure shift)
    shift = dot(cache.areah, ρ) / sum(cache.areah)

    # device: batched recovery of (u, p, L)
    copyto!(work.Λ_d, T.(Λn))
    copyto!(work.ρ_d, T.(ρ))
    _ns_gather!(backend)(work.lam, work.Λ_d, batch.elcon, nps; ndrange=(2 * nfc, nt))
    _ns_recover!(backend)(work.un, work.pn, work.Bhat, work.lam, work.ρ_d, npl,
                          ncB; ndrange=(nv, nt))
    _ns_gradient!(backend)(work.Ln, batch.MiE1, batch.MiE2, batch.MiCx,
                           batch.MiCy, work.lam, work.un, nfc; ndrange=(npl, nt))
    KernelAbstractions.synchronize(backend)

    un = Float64.(Array(work.un))
    pn = Float64.(Array(work.pn))
    Ln = Float64.(Array(work.Ln))
    pn .-= shift
    ρ .-= shift

    return (u=un, gradu=Ln, p=pn, Λ=Λn, ρ=ρ, cache=cache)
end

# ------------------------------------------------------------------------------
# Scalar transport (convection-diffusion) batch: HDGCDBatch + driver
# ------------------------------------------------------------------------------

"""
    HDGCDBatch(master, mesh, κ, τ; T=Float64)

Per-element constants of the batched scalar HDG transport step
([`hdg_cd_step`](@ref)): like [`HDGNSBatch`](@ref) but scalar-valued —
`A0 (npl, npl, nt)` and `B0 (npl, nfc + 1, nt)` carry the κ-viscous
composites and the constant −τ⟨θ̂, ·⟩ trace stabilization; the
state-dependent convection (volume `u`, trace λ) is added per call.
"""
struct HDGCDBatch{T, A2 <: AbstractMatrix{T}, A3 <: AbstractArray{T, 3},
                  I2 <: AbstractMatrix{Int32}, I3 <: AbstractArray{Int32, 3}}
    npl   :: Int
    nps   :: Int
    nfc   :: Int
    nt    :: Int
    ng    :: Int
    nq1d  :: Int
    shap  :: A2
    sh1d  :: A2
    shapx :: A3
    shapy :: A3
    M     :: A3
    A0    :: A3
    B0    :: A3
    Hx    :: A3
    Hlam0 :: A3
    MiE1  :: A3
    MiE2  :: A3
    MiCx  :: A3
    MiCy  :: A3
    wds   :: A3
    fn1   :: A3
    fn2   :: A3
    perm  :: I2
    elcon :: I3
end

Adapt.@adapt_structure HDGCDBatch

Base.eltype(::HDGCDBatch{T}) where {T} = T

function HDGCDBatch(master, mesh, κ, τ; T::Type{<:AbstractFloat}=Float64)
    nps = master.porder + 1
    nfc = 3 * nps
    npl = size(mesh.dgnodes, 1)
    nt = size(mesh.t, 1)
    ng = length(master.gwgh)
    nq1d = length(master.gw1d)
    ncB = nfc + 1

    shap2 = Matrix(master.shap[:, 1, :])
    sh1d = Matrix(master.sh1d[:, 1, :])

    shapx = zeros(npl, ng, nt)
    shapy = zeros(npl, ng, nt)
    M = zeros(npl, npl, nt)
    A0 = zeros(npl, npl, nt)
    B0 = zeros(npl, ncB, nt)
    Hx = zeros(nfc, npl, nt)
    Hlam0 = zeros(nfc, nfc, nt)
    MiE1 = zeros(npl, nfc, nt)
    MiE2 = zeros(npl, nfc, nt)
    MiCx = zeros(npl, npl, nt)
    MiCy = zeros(npl, npl, nt)
    wdsb = zeros(nq1d, 3, nt)
    fn1 = zeros(nq1d, 3, nt)
    fn2 = zeros(nq1d, 3, nt)

    @views Threads.@threads for it in 1:nt
        dg = mesh.dgnodes[:, :, it]
        vol = hdg_elem_volume(dg, master)
        shapx[:, :, it] .= vol.shapx
        shapy[:, :, it] .= vol.shapy
        M[:, :, it] .= vol.M

        E1 = zeros(npl, nfc)
        E2 = zeros(npl, nfc)
        FN1 = zeros(npl, npl)
        FN2 = zeros(npl, npl)
        Fτ = zeros(npl, npl)
        Bλ0 = zeros(npl, nfc)
        HN1 = zeros(nfc, npl)
        HN2 = zeros(nfc, npl)
        Hτ = zeros(nfc, npl)
        Hλf0 = zeros(nfc, nfc)

        for s in 1:3
            ed = hdg_elem_edge(dg, master, s)
            cols = (s - 1) * nps .+ (1:nps)
            T0 = facemat(sh1d, ed.wds)
            Tn1 = facemat(sh1d, ed.wds .* ed.n1)
            Tn2 = facemat(sh1d, ed.wds .* ed.n2)
            E1[ed.ps, cols] .+= Tn1
            E2[ed.ps, cols] .+= Tn2
            FN1[ed.ps, ed.ps] .+= Tn1
            FN2[ed.ps, ed.ps] .+= Tn2
            Fτ[ed.ps, ed.ps] .+= τ .* T0
            Bλ0[ed.ps, cols] .-= τ .* T0
            HN1[cols, ed.ps] .+= Tn1
            HN2[cols, ed.ps] .+= Tn2
            Hτ[cols, ed.ps] .+= τ .* T0
            Hλf0[cols, cols] .-= τ .* T0
            wdsb[:, s, it] .= ed.wds
            fn1[:, s, it] .= ed.n1
            fn2[:, s, it] .= ed.n2
        end

        MF = cholesky(Symmetric(Matrix(vol.M)))
        MiCx_ = MF \ Matrix(vol.Cx')
        MiCy_ = MF \ Matrix(vol.Cy')
        MiE1_ = MF \ E1
        MiE2_ = MF \ E2
        MiE1[:, :, it] .= MiE1_
        MiE2[:, :, it] .= MiE2_
        MiCx[:, :, it] .= MiCx_
        MiCy[:, :, it] .= MiCy_

        G1 = κ .* (vol.Cx' .- FN1)
        G2 = κ .* (vol.Cy' .- FN2)

        A0[:, :, it] .= Fτ .- (G1 * MiCx_ .+ G2 * MiCy_)
        B0[:, 1:nfc, it] .= G1 * MiE1_ .+ G2 * MiE2_ .+ Bλ0
        Hx[:, :, it] .= κ .* (HN1 * MiCx_ .+ HN2 * MiCy_) .+ Hτ
        Hlam0[:, :, it] .= .-κ .* (HN1 * MiE1_ .+ HN2 * MiE2_) .+ Hλf0
    end

    perm = Int32.(master.perm[:, :, 1])
    elcon = Int32.(mesh.elcon)

    return HDGCDBatch(npl, nps, nfc, nt, ng, nq1d,
                      T.(shap2), T.(sh1d), T.(shapx), T.(shapy), T.(M),
                      T.(A0), T.(B0), T.(Hx), T.(Hlam0),
                      T.(MiE1), T.(MiE2), T.(MiCx), T.(MiCy),
                      T.(wdsb), T.(fn1), T.(fn2), perm, elcon)
end

# Scalar trace pattern: BC *types* per face are decided by `tbc` and assumed
# fixed across calls (values may change); Dirichlet faces get identity rows,
# flux faces keep their continuity rows.
struct HDGCDTracePattern
    II    :: Vector{Int}
    JJ    :: Vector{Int}
    gdofs :: Matrix{Int}     # (nfc, nt)
    isdbc :: BitVector
    ndof  :: Int
end

function HDGCDTracePattern(mesh, master, tbc)
    nps = mesh.porder + 1
    nfc = 3 * nps
    nt = size(mesh.t, 1)
    nf = size(mesh.f, 1)
    ndof = nps * nf
    elcon = mesh.elcon

    isdbc = falses(ndof)
    for i in 1:nf
        mesh.f[i, 4] >= 0 && continue
        tag = -mesh.f[i, 4]
        Xq, _, _ = boundary_face_quad(mesh, master, i)
        if tbc(view(Xq, 1, :), tag)[1] == :d
            isdbc[(i - 1) * nps .+ (1:nps)] .= true
        end
    end

    gdofs = zeros(Int, nfc, nt)
    for it in 1:nt, s in 1:3, a in 1:nps
        gdofs[(s - 1) * nps + a, it] = elcon[a, s, it]
    end

    II = Int[]
    JJ = Int[]
    sizehint!(II, nt * nfc^2)
    sizehint!(JJ, nt * nfc^2)
    for it in 1:nt
        for jl in 1:nfc
            gj = gdofs[jl, it]
            for il in 1:nfc
                gi = gdofs[il, it]
                isdbc[gi] && continue
                push!(II, gi); push!(JJ, gj)
            end
        end
    end
    for gd in 1:ndof
        if isdbc[gd]
            push!(II, gd); push!(JJ, gd)
        end
    end

    return HDGCDTracePattern(II, JJ, gdofs, isdbc, ndof)
end

function cd_trace_fill!(VV, rhs, pat::HDGCDTracePattern, Kes, res, gvals, nrhs)
    nfc = size(Kes, 1)
    nt = size(Kes, 3)
    n = 0
    fill!(rhs, 0.0)
    @inbounds for it in 1:nt
        for jl in 1:nfc
            for il in 1:nfc
                gi = pat.gdofs[il, it]
                pat.isdbc[gi] && continue
                n += 1
                VV[n] = Kes[il, jl, it]
            end
        end
        for il in 1:nfc
            gi = pat.gdofs[il, it]
            pat.isdbc[gi] || (rhs[gi] += res[il, it])
        end
    end
    rhs .+= nrhs
    @inbounds for gd in 1:pat.ndof
        if pat.isdbc[gd]
            n += 1
            VV[n] = 1.0
            rhs[gd] = gvals[gd]
        end
    end
    @assert n == length(VV)
    return VV, rhs
end

mutable struct HDGCDCache{B <: HDGCDBatch, W}
    batch :: B
    pat   :: HDGCDTracePattern
    work  :: W
    pgh   :: Array{Float64, 3}
    wjach :: Matrix{Float64}
    shaph :: Matrix{Float64}
    VV    :: Vector{Float64}
    rhs   :: Vector{Float64}
    F     :: Any
    κ     :: Float64
    τ     :: Float64
end

function _cd_make_work(backend, T, npl, nfc, ng, nt, nΛ, ndof)
    ncB = nfc + 1
    kz(dims...) = KernelAbstractions.zeros(backend, T, dims...)
    return (; ug = kz(ng, 2, nt),
            K1 = kz(npl, npl, nt), K2 = kz(npl, npl, nt),
            A = kz(npl, npl, nt), B = kz(npl, ncB, nt),
            HxZ = kz(nfc, ncB, nt), Hlam = kz(nfc, nfc, nt),
            lam = kz(2 * nfc, nt), th = kz(nfc, nt),
            u_d = kz(npl, 2, nt), θold_d = kz(npl, nt),
            fsrc = kz(npl, nt), rext = kz(npl, nt),
            Λ_d = kz(nΛ), Θ_d = kz(ndof),
            θn = kz(npl, nt), qn = kz(npl, 2, nt))
end

function HDGCDCache(master, mesh, κ, τ, tbc; ArrayT=Array,
                    T::Type{<:AbstractFloat}=Float64)
    batch_h = HDGCDBatch(master, mesh, κ, τ; T)

    npl, ng, nt = batch_h.npl, batch_h.ng, batch_h.nt
    pgh = zeros(ng, 2, nt)
    wjach = zeros(ng, nt)
    shaph = Matrix(master.shap[:, 1, :])
    @views Threads.@threads for it in 1:nt
        vol = hdg_elem_volume(mesh.dgnodes[:, :, it], master)
        pgh[:, :, it] .= shaph' * mesh.dgnodes[:, :, it]
        wjach[:, it] .= vol.wjac
    end

    batch = adapt(ArrayT, batch_h)
    pat = HDGCDTracePattern(mesh, master, tbc)
    backend = KernelAbstractions.get_backend(batch.M)
    nΛ = 2 * batch.nps * size(mesh.f, 1)
    work = _cd_make_work(backend, T, batch.npl, batch.nfc, batch.ng, batch.nt,
                         nΛ, pat.ndof)
    return HDGCDCache(batch, pat, work, pgh, wjach, shaph,
                      zeros(length(pat.II)), zeros(pat.ndof), nothing,
                      Float64(κ), Float64(τ))
end

"""
    hdg_cd_step_batched(master, mesh, κ, tbc; τ=1.0, u=nothing, Λ=nothing,
                        θold=nothing, dtinv=0.0, source=nothing,
                        ArrayT=Array, T=Float64, cache=nothing)

Batched/KA counterpart of [`hdg_cd_step`](@ref) (same arguments, same returned
fields plus `cache`). The per-element assembly, local solves and (θ, q)
recovery run on the backend of `ArrayT`; the trace system is a CPU sparse LU
with the pattern and factorization reused across calls. The boundary-condition
*types* returned by `tbc` must not change between calls with the same cache.
"""
function hdg_cd_step_batched(master, mesh, κ, tbc; τ=1.0, u=nothing, Λ=nothing,
                             θold=nothing, dtinv=0.0, source=nothing,
                             ArrayT=Array, T::Type{<:AbstractFloat}=Float64,
                             cache=nothing)
    if cache === nothing
        cache = HDGCDCache(master, mesh, κ, τ, tbc; ArrayT, T)
    else
        (cache.κ == κ && cache.τ == τ) ||
            throw(ArgumentError("cache was built with κ=$(cache.κ), τ=$(cache.τ)"))
    end
    batch, work, pat = cache.batch, cache.work, cache.pat
    backend = KernelAbstractions.get_backend(batch.M)
    npl, nps, nfc, ng, nt = batch.npl, batch.nps, batch.nfc, batch.ng, batch.nt
    ncB = nfc + 1

    u === nothing ? fill!(work.u_d, zero(T)) : copyto!(work.u_d, T.(u))
    Λ === nothing ? fill!(work.Λ_d, zero(T)) : copyto!(work.Λ_d, T.(Λ))
    θold === nothing ? fill!(work.θold_d, zero(T)) : copyto!(work.θold_d, T.(θold))
    if source isa AbstractArray
        copyto!(work.fsrc, T.(source))
        fill!(work.rext, zero(T))
    elseif source isa Function
        fill!(work.fsrc, zero(T))
        rext = zeros(npl, nt)
        @views Threads.@threads for it in 1:nt
            sg = [source(cache.pgh[g, :, it]) for g in 1:ng]
            rext[:, it] .= cache.shaph * (cache.wjach[:, it] .* sg)
        end
        copyto!(work.rext, T.(rext))
    else
        fill!(work.fsrc, zero(T))
        fill!(work.rext, zero(T))
    end

    # device: local systems
    _ns_quadvel!(backend)(work.ug, batch.shap, work.u_d; ndrange=(ng, 2, nt))
    _ns_wgemm!(backend)(work.K1, batch.shapx, work.ug, 1, batch.shap; ndrange=(npl, npl, nt))
    _ns_wgemm!(backend)(work.K2, batch.shapy, work.ug, 2, batch.shap; ndrange=(npl, npl, nt))
    work.A .= batch.A0 .- work.K1 .- work.K2 .+ T(dtinv) .* batch.M
    copyto!(work.B, batch.B0)
    copyto!(work.Hlam, batch.Hlam0)
    _ns_gather!(backend)(work.lam, work.Λ_d, batch.elcon, nps; ndrange=(2 * nfc, nt))
    _cd_rhs!(backend)(work.B, batch.M, work.θold_d, work.fsrc, work.rext,
                      T(dtinv), ncB; ndrange=(npl, nt))
    _cd_faces!(backend)(work.B, work.Hlam, work.lam, batch.sh1d, batch.wds,
                        batch.fn1, batch.fn2, batch.perm, nps, ncB; ndrange=nt)
    _blusolve!(backend)(work.A, work.B; ndrange=nt)           # B := Z
    _bgemm_nn!(backend)(work.HxZ, batch.Hx, work.B, one(T), zero(T);
                        ndrange=(nfc, ncB, nt))
    work.Hlam .-= view(work.HxZ, :, 1:nfc, :)                 # Ke
    KernelAbstractions.synchronize(backend)

    Kes = Float64.(Array(work.Hlam))
    res = -Float64.(Array(work.HxZ[:, ncB, :]))

    # host: BC data (Dirichlet L2 projection / prescribed-flux lift) + solve
    gvals = zeros(pat.ndof)
    nrhs = zeros(pat.ndof)
    sh1d = @view master.sh1d[:, 1, :]
    for i in axes(mesh.f, 1)
        mesh.f[i, 4] >= 0 && continue
        tag = -mesh.f[i, 4]
        Xq, wds, Tm = boundary_face_quad(mesh, master, i)
        bc = [tbc(view(Xq, g, :), tag) for g in axes(Xq, 1)]
        vals = Float64[b[2] for b in bc]
        if bc[1][1] == :d
            gproj = Tm \ (sh1d * (wds .* vals))
            gvals[(i - 1) * nps .+ (1:nps)] .= gproj
        else
            nrhs[(i - 1) * nps .+ (1:nps)] .+= sh1d * (wds .* vals)
        end
    end
    cd_trace_fill!(cache.VV, cache.rhs, pat, Kes, res, gvals, nrhs)
    H = sparse(pat.II, pat.JJ, cache.VV, pat.ndof, pat.ndof)
    if cache.F === nothing
        cache.F = lu(H)
    else
        lu!(cache.F, H)
    end
    Θ = cache.F \ cache.rhs

    # device: batched recovery of (θ, q)
    copyto!(work.Θ_d, T.(Θ))
    _cd_gather!(backend)(work.th, work.Θ_d, batch.elcon, nps; ndrange=(nfc, nt))
    _cd_recover!(backend)(work.θn, work.B, work.th, ncB; ndrange=(npl, nt))
    _cd_gradq!(backend)(work.qn, batch.MiE1, batch.MiE2, batch.MiCx,
                        batch.MiCy, work.th, work.θn; ndrange=(npl, nt))
    KernelAbstractions.synchronize(backend)

    θn = Float64.(Array(work.θn))
    qn = Float64.(Array(work.qn))

    return (θ=θn, q=qn, Θ=Θ, cache=cache)
end
