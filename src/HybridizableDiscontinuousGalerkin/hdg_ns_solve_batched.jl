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

