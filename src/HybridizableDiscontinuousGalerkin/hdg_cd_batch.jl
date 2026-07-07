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
Dimension-generic like the NS batch.
"""
struct HDGCDBatch{T, A2 <: AbstractMatrix{T}, A3 <: AbstractArray{T, 3},
                  A4 <: AbstractArray{T, 4},
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
    B0    :: A3
    Hx    :: A3
    Hlam0 :: A3
    MiE   :: A4
    MiC   :: A4
    wds   :: A3
    fn    :: A4
    perm  :: I2
    elcon :: I3
end

Adapt.@adapt_structure HDGCDBatch

Base.eltype(::HDGCDBatch{T}) where {T} = T
Base.ndims(batch::HDGCDBatch) = size(batch.fn, 2)

function HDGCDBatch(master, mesh, κ, τ; T::Type{<:AbstractFloat}=Float64)
    Dim = ndims(master)
    nfe = Dim + 1
    nps = size(master.perm, 1)
    nfc = nfe * nps
    npl = size(mesh.dgnodes, 1)
    nt = size(mesh.t, 1)
    ng = length(master.gwgh)
    nqf = length(master.face.gwgh)
    ncB = nfc + 1

    shap2 = Matrix(master.shap[:, 1, :])
    shf = Matrix(master.face.shap[:, 1, :])

    shapd = zeros(npl, ng, Dim, nt)
    M = zeros(npl, npl, nt)
    A0 = zeros(npl, npl, nt)
    B0 = zeros(npl, ncB, nt)
    Hx = zeros(nfc, npl, nt)
    Hlam0 = zeros(nfc, nfc, nt)
    MiE = zeros(npl, nfc, Dim, nt)
    MiC = zeros(npl, npl, Dim, nt)
    wdsb = zeros(nqf, nfe, nt)
    fn = zeros(nqf, Dim, nfe, nt)

    @views Threads.@threads for it in 1:nt
        dg = mesh.dgnodes[:, :, it]
        vol = hdg_elem_volume(dg, master)
        shapd[:, :, :, it] .= vol.shapd
        M[:, :, it] .= vol.M

        E = zeros(npl, nfc, Dim)
        FN = zeros(npl, npl, Dim)
        Fτ = zeros(npl, npl)
        Bλ0 = zeros(npl, nfc)
        HN = zeros(nfc, npl, Dim)
        Hτ = zeros(nfc, npl)
        Hλf0 = zeros(nfc, nfc)

        for s in 1:nfe
            fc = hdg_elem_face(dg, master, s)
            cols = (s - 1) * nps .+ (1:nps)
            T0 = facemat(shf, fc.wds)
            for d in 1:Dim
                Tn = facemat(shf, fc.wds .* fc.n[:, d])
                E[fc.ps, cols, d] .+= Tn
                FN[fc.ps, fc.ps, d] .+= Tn
                HN[cols, fc.ps, d] .+= Tn
                fn[:, d, s, it] .= fc.n[:, d]
            end
            Fτ[fc.ps, fc.ps] .+= τ .* T0
            Bλ0[fc.ps, cols] .-= τ .* T0
            Hτ[cols, fc.ps] .+= τ .* T0
            Hλf0[cols, cols] .-= τ .* T0
            wdsb[:, s, it] .= fc.wds
        end

        MF = cholesky(Symmetric(Matrix(vol.M)))
        A0[:, :, it] .= Fτ
        B0[:, 1:nfc, it] .= Bλ0
        Hx[:, :, it] .= Hτ
        Hlam0[:, :, it] .= Hλf0
        for d in 1:Dim
            MiCd = MF \ Matrix(vol.C[:, :, d]')
            MiEd = MF \ E[:, :, d]
            MiE[:, :, d, it] .= MiEd
            MiC[:, :, d, it] .= MiCd
            Gd = κ .* (vol.C[:, :, d]' .- FN[:, :, d])
            A0[:, :, it] .-= Gd * MiCd
            B0[:, 1:nfc, it] .+= Gd * MiEd
            Hx[:, :, it] .+= κ .* (HN[:, :, d] * MiCd)
            Hlam0[:, :, it] .-= κ .* (HN[:, :, d] * MiEd)
        end
    end

    perm = Int32.(master.perm[:, :, 1])
    elcon = Int32.(mesh.elcon)

    return HDGCDBatch(npl, nps, nfc, nt, ng, nqf,
                      T.(shap2), T.(shf), T.(shapd), T.(M),
                      T.(A0), T.(B0), T.(Hx), T.(Hlam0),
                      T.(MiE), T.(MiC),
                      T.(wdsb), T.(fn), perm, elcon)
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
    Dim = ndims(master)
    nfe = Dim + 1
    nps = size(master.perm, 1)
    nfc = nfe * nps
    nt = size(mesh.t, 1)
    nf = size(mesh.f, 1)
    ndof = nps * nf
    elcon = mesh.elcon

    isdbc = falses(ndof)
    for i in 1:nf
        mesh.f[i, end] >= 0 && continue
        tag = -mesh.f[i, end]
        Xq, _, _ = boundary_face_quad(mesh, master, i)
        if tbc(view(Xq, 1, :), tag)[1] == :d
            isdbc[(i - 1) * nps .+ (1:nps)] .= true
        end
    end

    gdofs = zeros(Int, nfc, nt)
    for it in 1:nt, s in 1:nfe, a in 1:nps
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

"""
    HDGCDCache(master, mesh, κ, τ, tbc; ArrayT=Array, T=Float64)

Preallocated workspace for [`hdg_cd_step_batched`](@ref): the batched
element data (`batch::HDGCDBatch`, on the backend of `ArrayT`), the trace
sparsity pattern, the device work arrays of the local solves/recovery, the
host-side quadrature geometry used to evaluate function sources, and the
reused sparse-LU factorization of the trace system (`F`).

Built automatically on the first `hdg_cd_step_batched` call; pass the
returned cache back in on subsequent calls with the same `master`, `mesh`,
`κ`, `τ`, and boundary-condition types to skip all setup and reuse the
factorization (this reuse is what makes implicit time stepping cheap).
"""
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

function _cd_make_work(backend, T, npl, nfc, ng, nt, nΛ, ndof, Dim)
    ncB = nfc + 1
    kz(dims...) = KernelAbstractions.zeros(backend, T, dims...)
    return (; ug = kz(ng, Dim, nt),
            K = kz(npl, npl, Dim, nt),
            A = kz(npl, npl, nt), B = kz(npl, ncB, nt),
            HxZ = kz(nfc, ncB, nt), Hlam = kz(nfc, nfc, nt),
            lam = kz(Dim * nfc, nt), th = kz(nfc, nt),
            u_d = kz(npl, Dim, nt), θold_d = kz(npl, nt),
            fsrc = kz(npl, nt), rext = kz(npl, nt),
            Λ_d = kz(nΛ), Θ_d = kz(ndof),
            θn = kz(npl, nt), qn = kz(npl, Dim, nt))
end

function HDGCDCache(master, mesh, κ, τ, tbc; ArrayT=Array,
                    T::Type{<:AbstractFloat}=Float64)
    Dim = ndims(master)
    batch_h = HDGCDBatch(master, mesh, κ, τ; T)

    npl, ng, nt = batch_h.npl, batch_h.ng, batch_h.nt
    pgh = zeros(ng, Dim, nt)
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
    nΛ = Dim * batch.nps * size(mesh.f, 1)
    work = _cd_make_work(backend, T, batch.npl, batch.nfc, batch.ng, batch.nt,
                         nΛ, pat.ndof, Dim)
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
    Dim = ndims(batch)
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
    _ns_quadvel!(backend)(work.ug, batch.shap, work.u_d; ndrange=(ng, Dim, nt))
    for d in 1:Dim
        _ns_wgemm!(backend)(view(work.K, :, :, d, :),
                            view(batch.shapd, :, :, d, :), work.ug, d,
                            batch.shap; ndrange=(npl, npl, nt))
    end
    work.A .= batch.A0 .+ T(dtinv) .* batch.M
    for d in 1:Dim
        work.A .-= view(work.K, :, :, d, :)
    end
    copyto!(work.B, batch.B0)
    copyto!(work.Hlam, batch.Hlam0)
    _ns_gather!(backend)(work.lam, work.Λ_d, batch.elcon, nps, Dim;
                         ndrange=(Dim * nfc, nt))
    _cd_rhs!(backend)(work.B, batch.M, work.θold_d, work.fsrc, work.rext,
                      T(dtinv), ncB; ndrange=(npl, nt))
    _cd_faces!(backend)(work.B, work.Hlam, work.lam, batch.shf, batch.wds,
                        batch.fn, batch.perm, nps, ncB, Val(Dim); ndrange=nt)
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
    shf = @view master.face.shap[:, 1, :]
    for i in axes(mesh.f, 1)
        mesh.f[i, end] >= 0 && continue
        tag = -mesh.f[i, end]
        Xq, wds, Tm = boundary_face_quad(mesh, master, i)
        bc = [tbc(view(Xq, g, :), tag) for g in axes(Xq, 1)]
        vals = Float64[b[2] for b in bc]
        if bc[1][1] == :d
            gproj = Tm \ (shf * (wds .* vals))
            gvals[(i - 1) * nps .+ (1:nps)] .= gproj
        else
            nrhs[(i - 1) * nps .+ (1:nps)] .+= shf * (wds .* vals)
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
    _cd_gradq!(backend)(work.qn, batch.MiE, batch.MiC, work.th, work.θn;
                        ndrange=(npl, nt))
    KernelAbstractions.synchronize(backend)

    θn = Float64.(Array(work.θn))
    qn = Float64.(Array(work.qn))

    return (θ=θn, q=qn, Θ=Θ, cache=cache)
end
