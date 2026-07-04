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

