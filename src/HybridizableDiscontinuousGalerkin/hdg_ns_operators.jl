# Batched HDG Navier-Stokes and scalar-transport assembly on any KA backend
# (Phase 5 of GPU_PLAN.md; dimension-generic since THREED_PLAN Phase E3).
# The per-element math of `hdg_ns_elemmat` / `hdg_cd_elemmat` splits into
#
#   - geometry × (ν, τ) constants (mass, coupling, face lifts and the
#     eliminated-gradient viscous composites Avisc/Bvisc/Hvisc/Hλvisc),
#     built once on the CPU with the same helpers the reference path uses; and
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
`Adapt.@adapt_structure`). Dimension-generic: with `npl` volume nodes, `nps`
face nodes per face, `nfc = (Dim+1) nps`, the local solve size
`nv = (Dim+1) npl` (u₁, …, u_Dim, p) and `ncB = Dim·nfc + 2`
right-hand-side columns (trace columns + bρ + r):

- `shap (npl, ng)`, `shf (nps, nqf)`: shared shape values at quadrature.
- `shapd (npl, ng, Dim, nt)`: weighted physical derivative tables.
- `M (npl, npl, nt)`: element mass (the `dtinv·M` term is added per call).
- `A0 (nv, nv, nt)`: constant part of the local Newton matrix, including the
  viscous composite and the mean-pressure gauge row.
- `Bhat0 (nv, ncB, nt)`: constant part of the local RHS block `[B | bρ | r]`
  (`r` column zero).
- `Hx (Dim·nfc, nv, nt)`, `Hlam0 (Dim·nfc, Dim·nfc, nt)`: flux-continuity
  test blocks (fully constant, resp. constant part).
- `MiE (npl, nfc, Dim, nt)`, `MiC (npl, npl, Dim, nt)`: gradient-recovery
  maps `M⁻¹E_d`, `M⁻¹C_dᵀ`.
- `wds (nqf, Dim+1, nt)`, `fn (nqf, Dim, Dim+1, nt)`: face quadrature
  measure and unit normal per local face.
- `perm (nps, Dim+1)`, `elcon (nps, Dim+1, nt)` (`Int32`): face-to-volume
  node map and global face-node connectivity.
- `crow (Dim·nfc, nt)`, `area (nt)`: compatibility row and element measure
  (consumed on the host during global assembly).
"""
struct HDGNSBatch{T, A1 <: AbstractVector{T}, A2 <: AbstractMatrix{T},
                  A3 <: AbstractArray{T, 3}, A4 <: AbstractArray{T, 4},
                  I2 <: AbstractMatrix{Int32}, I3 <: AbstractArray{Int32, 3}}
    npl   :: Int
    nps   :: Int
    nfc   :: Int
    nt    :: Int
    ng    :: Int
    nqf   :: Int
    shap  :: A2
    shf   :: A2
    shapd :: A4
    M     :: A3
    A0    :: A3
    Bhat0 :: A3
    Hx    :: A3
    Hlam0 :: A3
    MiE   :: A4
    MiC   :: A4
    wds   :: A3
    fn    :: A4
    crow  :: A2
    area  :: A1
    perm  :: I2
    elcon :: I3
end

Adapt.@adapt_structure HDGNSBatch

Base.eltype(::HDGNSBatch{T}) where {T} = T
Base.ndims(batch::HDGNSBatch) = size(batch.fn, 2)

function HDGNSBatch(master, mesh, ν, τ; T::Type{<:AbstractFloat}=Float64)
    Dim = ndims(master)
    nfe = Dim + 1
    nps = size(master.perm, 1)
    nfc = nfe * nps
    npl = size(mesh.dgnodes, 1)
    nt = size(mesh.t, 1)
    ng = length(master.gwgh)
    nqf = length(master.face.gwgh)
    nv = (Dim + 1) * npl
    ncB = Dim * nfc + 2

    shap2 = Matrix(master.shap[:, 1, :])
    shf = Matrix(master.face.shap[:, 1, :])

    shapd = zeros(npl, ng, Dim, nt)
    M = zeros(npl, npl, nt)
    A0 = zeros(nv, nv, nt)
    Bhat0 = zeros(nv, ncB, nt)
    Hx = zeros(Dim * nfc, nv, nt)
    Hlam0 = zeros(Dim * nfc, Dim * nfc, nt)
    MiE = zeros(npl, nfc, Dim, nt)
    MiC = zeros(npl, npl, Dim, nt)
    wdsb = zeros(nqf, nfe, nt)
    fn = zeros(nqf, Dim, nfe, nt)
    crow = zeros(Dim * nfc, nt)
    area = zeros(nt)

    iu = ntuple(c -> (c - 1) * npl .+ (1:npl), Dim)
    ip = Dim * npl .+ (1:npl)
    jλ = ntuple(c -> (c - 1) * nfc .+ (1:nfc), Dim)
    icp = Dim * npl + 1

    @views Threads.@threads for it in 1:nt
        dg = mesh.dgnodes[:, :, it]
        vol = hdg_elem_volume(dg, master)
        shapd[:, :, :, it] .= vol.shapd
        M[:, :, it] .= vol.M
        area[it] = sum(vol.wjac)

        E = zeros(npl, nfc, Dim)
        FN = zeros(npl, npl, Dim)
        Fτ = zeros(npl, npl)
        Eτ = zeros(npl, nfc)
        HN = zeros(nfc, npl, Dim)
        Hτ = zeros(nfc, npl)
        Hλτ = zeros(nfc, nfc)

        for s in 1:nfe
            fc = hdg_elem_face(dg, master, s)
            cols = (s - 1) * nps .+ (1:nps)
            T0 = facemat(shf, fc.wds)
            for d in 1:Dim
                Tn = facemat(shf, fc.wds .* fc.n[:, d])
                E[fc.ps, cols, d] .+= Tn
                FN[fc.ps, fc.ps, d] .+= Tn
                HN[cols, fc.ps, d] .+= Tn
                crow[(d - 1) * nfc .+ cols, it] .+= shf * (fc.wds .* fc.n[:, d])
                fn[:, d, s, it] .= fc.n[:, d]
            end
            Fτ[fc.ps, fc.ps] .+= τ .* T0
            Eτ[fc.ps, cols] .+= τ .* T0
            Hτ[cols, fc.ps] .+= τ .* T0
            Hλτ[cols, cols] .+= τ .* T0
            wdsb[:, s, it] .= fc.wds
        end

        MF = cholesky(Symmetric(Matrix(vol.M)))
        Avisc = zeros(npl, npl)
        Bvisc = zeros(npl, nfc)
        Hvisc = zeros(nfc, npl)
        Hλvisc = zeros(nfc, nfc)
        for d in 1:Dim
            MiCd = MF \ Matrix(vol.C[:, :, d]')
            MiEd = MF \ E[:, :, d]
            MiE[:, :, d, it] .= MiEd
            MiC[:, :, d, it] .= MiCd
            Gd = ν .* (vol.C[:, :, d]' .- FN[:, :, d])
            Avisc .-= Gd * MiCd
            Bvisc .+= Gd * MiEd
            Hvisc .+= ν .* (HN[:, :, d] * MiCd)
            Hλvisc .-= ν .* (HN[:, :, d] * MiEd)
        end

        for c in 1:Dim
            A0[iu[c], iu[c], it] .= Avisc .+ Fτ
            A0[iu[c], ip, it] .= .-vol.C[:, :, c]' .+ FN[:, :, c]
            A0[ip, iu[c], it] .= .-vol.C[:, :, c]'
            Bhat0[iu[c], jλ[c], it] .= Bvisc .- Eτ
            Bhat0[ip, jλ[c], it] .= E[:, :, c]
            Hx[jλ[c], iu[c], it] .= Hvisc .+ Hτ
            Hx[jλ[c], ip, it] .= HN[:, :, c]
            Hlam0[jλ[c], jλ[c], it] .= Hλvisc .- Hλτ
        end
        A0[icp, :, it] .= 0.0
        A0[icp, ip, it] .= shap2 * vol.wjac
        Bhat0[icp, :, it] .= 0.0
        Bhat0[icp, Dim * nfc + 1, it] = -area[it]
    end

    perm = Int32.(master.perm[:, :, 1])
    elcon = Int32.(mesh.elcon)

    return HDGNSBatch(npl, nps, nfc, nt, ng, nqf,
                      T.(shap2), T.(shf), T.(shapd), T.(M),
                      T.(A0), T.(Bhat0), T.(Hx), T.(Hlam0),
                      T.(MiE), T.(MiC),
                      T.(wdsb), T.(fn), T.(crow), T.(area),
                      perm, elcon)
end
