# Batched HDG element assembly and local recovery on any KA backend
# (Phase 3b of GPU_PLAN.md; dimension-generic since THREED_PLAN Phase E).
# The per-element math of `localprob`/`elemmat_hdg` is restated as batched
# dense linear algebra, with the spatial direction as an array axis (D5):
#
#   local solves:  K [U | u0] = [-Lu + Σ_d C_d M⁻¹ L_d | F_src],
#                  K = D + Σ_d C_d M⁻¹ C_d',
#                  Q_d = (M⁻¹ C_d') U - M⁻¹ L_d
#   assembly:      ae = -(Aλ + Σ_d R_d Q_d + Ru U),
#                  fe =  Σ_d R_d q0_d + Ru u0
#   recovery:      uh = U m + u0,  qh = Q m + q0   (m = element trace values)
#
# The small per-element matrices (M⁻¹, C_d, D, face lifts L_d, trace-test
# matrices R_d/Ru/Aλ) are geometry × parameters, built once on the CPU
# (reusing the canonical Geometry caches); the batched GEMMs and the
# per-element LU solve run as KA kernels. The unit-trace solutions U/Q_d
# double as the trace-to-solution maps, so recovery is a batched matvec
# instead of a per-element re-solve.
#
# Faithful to the legacy 2D code in two easily-missed ways: `localprob` uses
# taud = κ for its internal stabilization while `elemmat_hdg` uses
# param[:taud] in the numerical flux, and nodes shared by several faces
# (corners in 2D; edges and vertices in 3D) accumulate face-lift
# contributions from every adjacent face.

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using Adapt
using LinearAlgebra
using SparseArrays
using TwoDG.Geometry: GeometricFactors, SideGeometry
using TwoDG.Meshes: mkelcon

"""
    HDGBatch(master, mesh, source, param; T=Float64)

Per-element operator matrices for the batched HDG assembly/recovery path,
built once on the CPU. All fields are plain arrays (`Adapt.@adapt_structure`),
so `adapt(CuArray, batch)` moves everything to the GPU. Dimension-generic:
`Dim` is the direction axis of `C`/`Le`/`R` (2 on triangles, 3 on
tetrahedra).

Shapes (`npl` volume nodes, `nps` trace nodes per face — `porder + 1` in 2D,
`(porder+1)(porder+2)/2` in 3D — `ndf = (Dim+1) nps` trace DOFs per element,
`nc1 = ndf + 1` local-solve columns — the unit-trace columns plus the source
column):

- `MinvM (npl, npl, nt)`: κ × inverse mass (the inverse of `localprob`'s `M`).
- `C (npl, npl, Dim, nt)`, `D (npl, npl, nt)`: coupling matrices and
  convection + stabilization operator (stabilization with `tau = κ + |c·n|`,
  as in `localprob`).
- `Le (npl, nc1, Dim, nt)`: face lifts of the trace (`F_d = -Le_d m`), last
  column zero so the source column threads through the same GEMMs.
- `B0 (npl, nc1, nt)`: local-solve RHS seed `[-Lu | F_src]`.
- `Alam (ndf, ndf, nt)`, `R (ndf, npl, Dim, nt)`, `Ru (ndf, npl, nt)`:
  trace-test matrices of the numerical flux (`tau = param[:taud] + |c·n|`,
  as in `elemmat_hdg`).
- `elcon (ndf, nt)`: element trace DOFs -> global trace DOFs (orientation
  resolved through the mesh's [`mkelcon`](@ref TwoDG.Meshes.mkelcon)
  connectivity — reversal in 2D, the 6 triangle symmetries in 3D), for the
  assembly scatter and recovery gather.
"""
struct HDGBatch{T, A3 <: AbstractArray{T, 3}, A4 <: AbstractArray{T, 4},
                I2 <: AbstractMatrix{Int32}}
    npl   :: Int
    nps   :: Int
    ndf   :: Int
    nt    :: Int
    MinvM :: A3
    C     :: A4
    D     :: A3
    Le    :: A4
    B0    :: A3
    Alam  :: A3
    R     :: A4
    Ru    :: A3
    elcon :: I2
end

Adapt.@adapt_structure HDGBatch

Base.eltype(::HDGBatch{T}) where {T} = T
Base.ndims(batch::HDGBatch) = size(batch.C, 3)

# element trace DOFs -> global trace DOFs in local-face-block order (ndf, nt):
# face f owns nodes (f-1)nps+1 : f*nps in its canonical order; the mesh's
# `elcon` already resolves each element's traversal through `orient_perm`
function _hdg_elcon(mesh, nps, nfe)
    elc = mesh.elcon === nothing ?
          mkelcon(mesh.t2f, mesh.t2o, mesh.porder) : mesh.elcon
    (size(elc, 1) == nps && size(elc, 2) == nfe) ||
        throw(DimensionMismatch("mesh.elcon is $(size(elc)); expected ($nps, $nfe, nt)"))
    return Int32.(reshape(elc, nps * nfe, size(elc, 3)))
end

function HDGBatch(master, mesh, source, param; T::Type{<:AbstractFloat}=Float64)
    Dim = ndims(master)
    kappa = param[:kappa]
    c = param[:c]
    length(c) == Dim ||
        throw(ArgumentError("param[:c] has $(length(c)) components; the mesh is $(Dim)D"))
    taud_ae = param[:taud]

    nfe = Dim + 1
    nps = size(master.perm, 1)
    ndf = nfe * nps
    nc1 = ndf + 1
    npl = size(mesh.dgnodes, 1)
    nt = size(mesh.t, 1)

    ctx = GeometricFactors(master, mesh)   # canonical volume geometry (Float64)
    side = SideGeometry(master, mesh)      # face metrics in element orientation
    shap = ctx.shap                        # (npl, ng)
    fshap = master.face.shap[:, 1, :]      # (nps, ngf) face trace basis
    perm = master.perm[:, :, 1]

    MinvM = zeros(npl, npl, nt)
    C = zeros(npl, npl, Dim, nt)
    D = zeros(npl, npl, nt)
    Le = zeros(npl, nc1, Dim, nt)
    B0 = zeros(npl, nc1, nt)
    Alam = zeros(ndf, ndf, nt)
    R = zeros(ndf, npl, Dim, nt)
    Ru = zeros(ndf, npl, nt)

    @views Threads.@threads for e in 1:nt
        MinvM[:, :, e] .= kappa .* ctx.Minv[:, :, e]
        for d in 1:Dim
            C[:, :, d, e] .= shap * ctx.shapd[:, :, d, e]'
            D[:, :, e] .-= c[d] .* C[:, :, d, e]'
        end

        if source isa Function
            src = source(ctx.pg[:, :, e])
            B0[:, nc1, e] .= shap * (ctx.wjac[:, e] .* vec(src))
        end

        for s in 1:nfe
            ps = perm[:, s]
            sw = side.sw[:, s, e]
            cnl = c[1] .* side.nl[:, 1, s, e]
            for d in 2:Dim
                cnl .+= c[d] .* side.nl[:, d, s, e]
            end
            tau_loc = kappa .+ abs.(cnl)      # localprob's tau (taud = kappa)
            tau_ae = taud_ae .+ abs.(cnl)     # elemmat_hdg's tau

            cols = (s-1)*nps+1 : s*nps
            for d in 1:Dim
                Ed = fshap * Diagonal(sw .* side.nl[:, d, s, e]) * fshap'
                Le[ps, cols, d, e] .+= Ed
                R[cols, ps, d, e] .+= Ed
            end
            Eu = fshap * Diagonal(sw .* (cnl .- tau_loc)) * fshap'
            Wu = fshap * Diagonal(sw .* tau_ae) * fshap'
            Aλ = fshap * Diagonal(sw .* (cnl .- tau_ae)) * fshap'

            D[ps, ps, e] .+= fshap * Diagonal(sw .* tau_loc) * fshap'
            B0[ps, cols, e] .-= Eu
            Ru[cols, ps, e] .+= Wu
            Alam[cols, cols, e] .+= Aλ
        end
    end

    elcon = _hdg_elcon(mesh, nps, nfe)

    return HDGBatch(npl, nps, ndf, nt,
                    T.(MinvM), T.(C), T.(D), T.(Le), T.(B0),
                    T.(Alam), T.(R), T.(Ru), elcon)
end

# C[:,:,e] = α A[:,:,e] * B[:,:,e] + β C[:,:,e], one workitem per entry
@kernel function _bgemm_nn!(C, @Const(A), @Const(B), α, β)
    i, j, e = @index(Global, NTuple)
    T = eltype(C)
    K = size(A, 2)
    acc = zero(T)
    @inbounds for k in 1:K
        acc += A[i, k, e] * B[k, j, e]
    end
    @inbounds C[i, j, e] = β == zero(T) ? α * acc : α * acc + β * C[i, j, e]
end

# C[:,:,e] = α A[:,:,e] * B[:,:,e]' + β C[:,:,e]
@kernel function _bgemm_nt!(C, @Const(A), @Const(B), α, β)
    i, j, e = @index(Global, NTuple)
    T = eltype(C)
    K = size(A, 2)
    acc = zero(T)
    @inbounds for k in 1:K
        acc += A[i, k, e] * B[j, k, e]
    end
    @inbounds C[i, j, e] = β == zero(T) ? α * acc : α * acc + β * C[i, j, e]
end

# In-place batched dense solve K[:,:,e] \ X[:,:,e]: Gaussian elimination with
# partial pivoting, one workitem per element (K is destroyed). Sizes are tiny
# (npl ≤ ~35), so a serial per-element factorization is appropriate.
@kernel function _blusolve!(K, X)
    e = @index(Global)
    T = eltype(K)
    n = size(K, 1)
    m = size(X, 2)

    @inbounds for k in 1:n
        p = k
        amax = abs(K[k, k, e])
        for i in k+1:n
            a = abs(K[i, k, e])
            if a > amax
                amax = a
                p = i
            end
        end
        if p != k
            for j in k:n
                tmp = K[k, j, e]; K[k, j, e] = K[p, j, e]; K[p, j, e] = tmp
            end
            for j in 1:m
                tmp = X[k, j, e]; X[k, j, e] = X[p, j, e]; X[p, j, e] = tmp
            end
        end
        kk = K[k, k, e]
        for i in k+1:n
            f = K[i, k, e] / kk
            if f != zero(T)
                for j in k+1:n
                    K[i, j, e] -= f * K[k, j, e]
                end
                for j in 1:m
                    X[i, j, e] -= f * X[k, j, e]
                end
            end
        end
    end
    @inbounds for k in n:-1:1
        kk = K[k, k, e]
        for j in 1:m
            acc = X[k, j, e]
            for i in k+1:n
                acc -= K[k, i, e] * X[i, j, e]
            end
            X[k, j, e] = acc / kk
        end
    end
end

@kernel function _gather_trace!(m, @Const(x), @Const(elcon))
    k, e = @index(Global, NTuple)
    @inbounds m[k, e] = x[elcon[k, e]]
end

# uh = U[:,1:ndf] m + U[:,end]; q_d likewise from Q (npl, nc1, Dim, nt),
# one workitem per node
@kernel function _recover!(u, q, @Const(U), @Const(Q), @Const(m))
    i, e = @index(Global, NTuple)
    ndf = size(m, 1)
    Dim = size(q, 2)
    accu = U[i, ndf+1, e]
    @inbounds for k in 1:ndf
        accu += U[i, k, e] * m[k, e]
    end
    @inbounds u[i, e] = accu
    @inbounds for d in 1:Dim
        acc = Q[i, ndf+1, d, e]
        for k in 1:ndf
            acc += Q[i, k, d, e] * m[k, e]
        end
        q[i, d, e] = acc
    end
end

"""
    hdg_local_solves(batch::HDGBatch)

Runs the batched local solves and static condensation on the backend of
`batch`. Returns `(; ae, fe, U, Q)`: the element matrices/vectors
`ae (ndf, ndf, nt)`, `fe (ndf, nt)` (Dirichlet BCs *not* yet applied) and the
local solution maps `U (npl, ndf + 1, nt)` / `Q (npl, ndf + 1, Dim, nt)`
(unit-trace columns plus the source column) consumed by
[`hdg_recover`](@ref).
"""
function hdg_local_solves(batch::HDGBatch{T}) where {T}
    backend = KernelAbstractions.get_backend(batch.D)
    npl, ndf, nt = batch.npl, batch.ndf, batch.nt
    Dim = ndims(batch)
    nc1 = ndf + 1
    z = zero(T)
    o = one(T)

    nn = _bgemm_nn!(backend)
    ntk = _bgemm_nt!(backend)

    Td = KernelAbstractions.zeros(backend, T, npl, nc1, Dim, nt)   # M⁻¹ L_d
    Xd = KernelAbstractions.zeros(backend, T, npl, npl, Dim, nt)   # M⁻¹ C_d'
    for d in 1:Dim
        nn(view(Td, :, :, d, :), batch.MinvM, view(batch.Le, :, :, d, :), o, z;
           ndrange=(npl, nc1, nt))
        ntk(view(Xd, :, :, d, :), batch.MinvM, view(batch.C, :, :, d, :), o, z;
            ndrange=(npl, npl, nt))
    end

    K = copy(batch.D)
    for d in 1:Dim
        nn(K, view(batch.C, :, :, d, :), view(Xd, :, :, d, :), o, o;
           ndrange=(npl, npl, nt))
    end

    U = copy(batch.B0)
    for d in 1:Dim
        nn(U, view(batch.C, :, :, d, :), view(Td, :, :, d, :), o, o;
           ndrange=(npl, nc1, nt))
    end
    _blusolve!(backend)(K, U; ndrange=nt)                     # U := K \ RHS

    Q = Td; Q .= .-Td                                         # reuse buffer
    for d in 1:Dim
        nn(view(Q, :, :, d, :), view(Xd, :, :, d, :), U, o, o;
           ndrange=(npl, nc1, nt))
    end

    Z = KernelAbstractions.zeros(backend, T, ndf, nc1, nt)
    for d in 1:Dim
        nn(Z, view(batch.R, :, :, d, :), view(Q, :, :, d, :), o, o;
           ndrange=(ndf, nc1, nt))
    end
    nn(Z, batch.Ru, U, o, o; ndrange=(ndf, nc1, nt))

    ae = similar(Z, ndf, ndf, nt)
    ae .= .-(batch.Alam .+ view(Z, :, 1:ndf, :))
    fe = Z[:, nc1, :]
    KernelAbstractions.synchronize(backend)

    return (; ae, fe, U, Q)
end

"""
    hdg_recover(batch, loc, x) -> (uh, qh)

Recovers `uh (npl, nt)` and `qh (npl, Dim, nt)` from the global trace vector
`x` (on the same backend as `batch`) using the local solution maps from
[`hdg_local_solves`](@ref) — a batched matvec, no per-element re-solve.
"""
function hdg_recover(batch::HDGBatch{T}, loc, x) where {T}
    backend = KernelAbstractions.get_backend(batch.D)
    npl, ndf, nt = batch.npl, batch.ndf, batch.nt
    Dim = ndims(batch)

    m = KernelAbstractions.zeros(backend, T, ndf, nt)
    _gather_trace!(backend)(m, x, batch.elcon; ndrange=(ndf, nt))

    uh = KernelAbstractions.zeros(backend, T, npl, nt)
    qh = KernelAbstractions.zeros(backend, T, npl, Dim, nt)
    _recover!(backend)(uh, qh, loc.U, loc.Q, m; ndrange=(npl, nt))
    KernelAbstractions.synchronize(backend)

    return uh, qh
end

"""
    hdg_parsolve_batched(master, mesh, source, dbc, param;
                         ArrayT=Array, T=Float64, kwargs...)

Fully device-resident counterpart of [`hdg_parsolve`](@ref): batched element
assembly ([`hdg_local_solves`](@ref)) and local recovery
([`hdg_recover`](@ref)) run on the backend of `ArrayT` alongside the
[`hdg_gmres_ka`](@ref) trace solve. Only the Dirichlet-BC application and the
face-block global assembly (`hdg_densesystem`) remain on the CPU. `kwargs` are
forwarded to `hdg_gmres_ka`. Works on triangles and tetrahedra alike.

Returns `(uh, qh, uhath, niter)` with the same shapes as `hdg_parsolve`
(`uh (npl, 1, nt)`, `qh (npl, Dim, nt)`, `uhath (nps, nf)`).
"""
function hdg_parsolve_batched(master, mesh, source, dbc, param;
                              ArrayT=Array, T::Type{<:AbstractFloat}=Float64,
                              kwargs...)
    batch = adapt(ArrayT, HDGBatch(master, mesh, source, param; T))
    loc = hdg_local_solves(batch)

    ae = Float64.(Array(loc.ae))
    fe = Float64.(Array(loc.fe))
    hdg_applydbc!(ae, fe, master, mesh, dbc)

    sys = adapt(ArrayT, HDGSystem(ae, fe, mesh; T))
    x, stats = hdg_gmres_ka(sys; kwargs...)

    return _hdg_batched_solution(batch, loc, x)..., stats.niter
end

"""
    hdg_direct_batched(master, mesh, source, dbc, param; T=Float64)

Direct (sparse LU/Cholesky-free) counterpart of [`hdg_parsolve_batched`](@ref):
the *same* batched element assembly ([`hdg_local_solves`](@ref)) and local
recovery, with the condensed trace system assembled to a `SparseMatrixCSC`
and factorized instead of solved iteratively — the Direct/GMRES choice is an
algorithm choice over one assembly engine, not a separate code path.

Returns `(uh, qh, uhath)` with the same shapes as `hdg_parsolve_batched`.
"""
function hdg_direct_batched(master, mesh, source, dbc, param;
                            T::Type{<:AbstractFloat}=Float64)
    batch = HDGBatch(master, mesh, source, param; T)
    loc = hdg_local_solves(batch)

    ae = Float64.(Array(loc.ae))
    fe = Float64.(Array(loc.fe))
    hdg_applydbc!(ae, fe, master, mesh, dbc)

    K, F = hdg_trace_system(ae, fe, batch.elcon)
    x = K \ F

    return _hdg_batched_solution(batch, loc, T.(x))
end

"""
    hdg_trace_system(ae, fe, elcon) -> (K, F)

Assemble the global condensed trace system from the element trace blocks
`ae (ndf, ndf, nt)`, `fe (ndf, nt)` via the orientation-resolved trace
connectivity `elcon (ndf, nt)`: triplet-based sparse assembly (duplicate
entries sum, exactly like the matrix-free face matvec).
"""
function hdg_trace_system(ae::AbstractArray{Tv, 3}, fe, elcon) where {Tv}
    ndf, _, nt = size(ae)
    n = maximum(elcon)

    Iv = Vector{Int}(undef, ndf * ndf * nt)
    Jv = Vector{Int}(undef, ndf * ndf * nt)
    Vv = Vector{Tv}(undef, ndf * ndf * nt)
    idx = 1
    for e in 1:nt, j in 1:ndf, i in 1:ndf
        Iv[idx] = elcon[i, e]
        Jv[idx] = elcon[j, e]
        Vv[idx] = ae[i, j, e]
        idx += 1
    end
    K = sparse(Iv, Jv, Vv, n, n)

    F = zeros(Tv, n)
    for e in 1:nt, i in 1:ndf
        F[elcon[i, e]] += fe[i, e]
    end
    return K, F
end

# recover (uh, qh, uhath) in the standard output shapes from the trace vector
function _hdg_batched_solution(batch, loc, x)
    uh_d, qh_d = hdg_recover(batch, loc, x)
    uh = Array{Float64}(undef, batch.npl, 1, batch.nt)
    uh[:, 1, :] .= Array(uh_d)
    qh = Float64.(Array(qh_d))
    uhath = reshape(Float64.(Array(x)), batch.nps, :)
    return uh, qh, uhath
end
