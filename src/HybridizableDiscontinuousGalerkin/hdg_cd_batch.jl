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
