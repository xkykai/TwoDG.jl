# Batched HDG element assembly and local recovery on any KA backend
# (Phase 3b of GPU_PLAN.md). The per-element math of `localprob`/`elemmat_hdg`
# is restated as batched dense linear algebra:
#
#   local solves:  K [U | u0] = [-Lu + Cx M⁻¹ Lx + Cy M⁻¹ Ly | F_src],
#                  K = D + Cx M⁻¹ Cx' + Cy M⁻¹ Cy',
#                  Qx = (M⁻¹ Cx') U - M⁻¹ Lx   (Qy analogous)
#   assembly:      ae = -(Aλ + Rx Qx + Ry Qy + Ru U),
#                  fe =  Rx q0x + Ry q0y + Ru u0
#   recovery:      uh = U m + u0,  qh = Q m + q0   (m = element trace values)
#
# The small per-element matrices (M⁻¹, Cx, Cy, D, edge lifts L*, trace-test
# matrices R*/Aλ) are geometry × parameters, built once on the CPU (reusing
# DGContext geometry); the batched GEMMs and the per-element LU solve run as
# KA kernels. The unit-trace solutions U/Qx/Qy double as the trace-to-solution
# maps, so recovery is a batched matvec instead of a per-element re-solve.
#
# Faithful to the legacy code in two easily-missed ways: `localprob` uses
# taud = κ for its internal stabilization while `elemmat_hdg` uses
# param[:taud] in the numerical flux, and corner nodes accumulate edge-lift
# contributions from both adjacent edges.

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using Adapt
using LinearAlgebra
using TwoDG.DiscontinuousGalerkin: DGContext

"""
    HDGBatch(master, mesh, source, param; T=Float64)

Per-element operator matrices for the batched HDG assembly/recovery path,
built once on the CPU. All fields are plain arrays (`Adapt.@adapt_structure`),
so `adapt(CuArray, batch)` moves everything to the GPU.

Shapes (`npl` volume nodes, `nps = porder + 1` trace nodes per edge,
`ndf = 3nps` trace DOFs per element, `nc1 = ndf + 1` local-solve columns —
the unit-trace columns plus the source column):

- `MinvM (npl, npl, nt)`: κ × inverse mass (the inverse of `localprob`'s `M`).
- `Cx, Cy, D (npl, npl, nt)`: coupling matrices and convection + stabilization
  operator (stabilization with `tau = κ + |c·n|`, as in `localprob`).
- `Lxe, Lye (npl, nc1, nt)`: edge lifts of the trace (`Fx = -Lxe m`), last
  column zero so the source column threads through the same GEMMs.
- `B0 (npl, nc1, nt)`: local-solve RHS seed `[-Lu | F_src]`.
- `Alam (ndf, ndf, nt)`, `Rx, Ry, Ru (ndf, npl, nt)`: trace-test matrices of
  the numerical flux (`tau = param[:taud] + |c·n|`, as in `elemmat_hdg`).
- `elcon (ndf, nt)`: element trace DOFs -> global trace DOFs (orientation
  resolved), for the recovery gather.
"""
struct HDGBatch{T, A3 <: AbstractArray{T, 3}, I2 <: AbstractMatrix{Int32}}
    npl   :: Int
    nps   :: Int
    ndf   :: Int
    nt    :: Int
    MinvM :: A3
    Cx    :: A3
    Cy    :: A3
    D     :: A3
    Lxe   :: A3
    Lye   :: A3
    B0    :: A3
    Alam  :: A3
    Rx    :: A3
    Ry    :: A3
    Ru    :: A3
    elcon :: I2
end

Adapt.@adapt_structure HDGBatch

Base.eltype(::HDGBatch{T}) where {T} = T

function HDGBatch(master, mesh, source, param; T::Type{<:AbstractFloat}=Float64)
    kappa = param[:kappa]
    c = param[:c]
    taud_ae = param[:taud]

    nps = master.porder + 1
    ndf = 3 * nps
    nc1 = ndf + 1
    npl = size(mesh.dgnodes, 1)
    nt = size(mesh.t, 1)

    ctx = DGContext(master, mesh)   # canonical volume geometry (Float64)
    shap = ctx.shap                 # (npl, ng)
    sh1d = master.sh1d[:, 1, :]     # (nps, ng1d)
    sh1dξ = master.sh1d[:, 2, :]
    gw1d = master.gw1d
    perm = master.perm[:, :, 1]

    MinvM = zeros(npl, npl, nt)
    Cx = zeros(npl, npl, nt)
    Cy = zeros(npl, npl, nt)
    D = zeros(npl, npl, nt)
    Lxe = zeros(npl, nc1, nt)
    Lye = zeros(npl, nc1, nt)
    B0 = zeros(npl, nc1, nt)
    Alam = zeros(ndf, ndf, nt)
    Rx = zeros(ndf, npl, nt)
    Ry = zeros(ndf, npl, nt)
    Ru = zeros(ndf, npl, nt)

    @views Threads.@threads for e in 1:nt
        dg = mesh.dgnodes[:, :, e]

        MinvM[:, :, e] .= kappa .* ctx.Minv[:, :, e]
        Cx[:, :, e] .= shap * ctx.shapx[:, :, e]'
        Cy[:, :, e] .= shap * ctx.shapy[:, :, e]'
        D[:, :, e] .= .-c[1] .* Cx[:, :, e]' .- c[2] .* Cy[:, :, e]'

        if source isa Function
            src = source(ctx.pg[:, :, e])
            B0[:, nc1, e] .= shap * (ctx.wjac[:, e] .* vec(src))
        end

        for s in 1:3
            ps = perm[:, s]
            xxi = sh1dξ' * dg[ps, 1]
            yxi = sh1dξ' * dg[ps, 2]
            dsdxi = sqrt.(xxi .^ 2 .+ yxi .^ 2)
            nl1 = yxi ./ dsdxi
            nl2 = .-xxi ./ dsdxi
            cnl = c[1] .* nl1 .+ c[2] .* nl2
            sw = gw1d .* dsdxi
            tau_loc = kappa .+ abs.(cnl)      # localprob's tau (taud = kappa)
            tau_ae = taud_ae .+ abs.(cnl)     # elemmat_hdg's tau

            Ex = sh1d * Diagonal(sw .* nl1) * sh1d'
            Ey = sh1d * Diagonal(sw .* nl2) * sh1d'
            Eu = sh1d * Diagonal(sw .* (cnl .- tau_loc)) * sh1d'
            Wu = sh1d * Diagonal(sw .* tau_ae) * sh1d'
            Aλ = sh1d * Diagonal(sw .* (cnl .- tau_ae)) * sh1d'

            D[ps, ps, e] .+= sh1d * Diagonal(sw .* tau_loc) * sh1d'

            cols = (s-1)*nps+1 : s*nps
            Lxe[ps, cols, e] .+= Ex
            Lye[ps, cols, e] .+= Ey
            B0[ps, cols, e] .-= Eu

            Rx[cols, ps, e] .+= Ex
            Ry[cols, ps, e] .+= Ey
            Ru[cols, ps, e] .+= Wu
            Alam[cols, cols, e] .+= Aλ
        end
    end

    # element trace DOFs -> global trace DOFs (same as hdg_localrecovery)
    elcon = zeros(Int32, ndf, nt)
    for e in 1:nt, j in 1:3
        f = mesh.t2f[e, j]
        if f > 0
            elcon[(j-1)*nps+1:j*nps, e] .= (f-1)*nps+1:f*nps
        else
            f = abs(f)
            elcon[(j-1)*nps+1:j*nps, e] .= f*nps:-1:(f-1)*nps+1
        end
    end

    return HDGBatch(npl, nps, ndf, nt,
                    T.(MinvM), T.(Cx), T.(Cy), T.(D),
                    T.(Lxe), T.(Lye), T.(B0),
                    T.(Alam), T.(Rx), T.(Ry), T.(Ru), elcon)
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
# (npl ≤ ~28), so a serial per-element factorization is appropriate.
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

# uh = U[:,1:ndf] m + U[:,end] (and the same for qx, qy), one workitem per node
@kernel function _recover!(u, qx, qy, @Const(U), @Const(Qx), @Const(Qy), @Const(m))
    i, e = @index(Global, NTuple)
    ndf = size(m, 1)
    accu = U[i, ndf+1, e]
    accx = Qx[i, ndf+1, e]
    accy = Qy[i, ndf+1, e]
    @inbounds for k in 1:ndf
        mk = m[k, e]
        accu += U[i, k, e] * mk
        accx += Qx[i, k, e] * mk
        accy += Qy[i, k, e] * mk
    end
    @inbounds u[i, e] = accu
    @inbounds qx[i, e] = accx
    @inbounds qy[i, e] = accy
end

"""
    hdg_local_solves(batch::HDGBatch)

Runs the batched local solves and static condensation on the backend of
`batch`. Returns `(; ae, fe, U, Qx, Qy)`: the element matrices/vectors
`ae (ndf, ndf, nt)`, `fe (ndf, nt)` (Dirichlet BCs *not* yet applied) and the
local solution maps `U/Qx/Qy (npl, ndf + 1, nt)` (unit-trace columns plus the
source column) consumed by [`hdg_recover`](@ref).
"""
function hdg_local_solves(batch::HDGBatch{T}) where {T}
    backend = KernelAbstractions.get_backend(batch.D)
    npl, ndf, nt = batch.npl, batch.ndf, batch.nt
    nc1 = ndf + 1
    z = zero(T)
    o = one(T)

    nn = _bgemm_nn!(backend)
    ntk = _bgemm_nt!(backend)

    T1 = KernelAbstractions.zeros(backend, T, npl, nc1, nt)   # M⁻¹ Lx
    T2 = KernelAbstractions.zeros(backend, T, npl, nc1, nt)   # M⁻¹ Ly
    X1 = KernelAbstractions.zeros(backend, T, npl, npl, nt)   # M⁻¹ Cx'
    X2 = KernelAbstractions.zeros(backend, T, npl, npl, nt)   # M⁻¹ Cy'
    nn(T1, batch.MinvM, batch.Lxe, o, z; ndrange=(npl, nc1, nt))
    nn(T2, batch.MinvM, batch.Lye, o, z; ndrange=(npl, nc1, nt))
    ntk(X1, batch.MinvM, batch.Cx, o, z; ndrange=(npl, npl, nt))
    ntk(X2, batch.MinvM, batch.Cy, o, z; ndrange=(npl, npl, nt))

    K = copy(batch.D)
    nn(K, batch.Cx, X1, o, o; ndrange=(npl, npl, nt))
    nn(K, batch.Cy, X2, o, o; ndrange=(npl, npl, nt))

    U = copy(batch.B0)
    nn(U, batch.Cx, T1, o, o; ndrange=(npl, nc1, nt))
    nn(U, batch.Cy, T2, o, o; ndrange=(npl, nc1, nt))
    _blusolve!(backend)(K, U; ndrange=nt)                     # U := K \ RHS

    Qx = T1; Qx .= .-T1                                       # reuse buffers
    Qy = T2; Qy .= .-T2
    nn(Qx, X1, U, o, o; ndrange=(npl, nc1, nt))
    nn(Qy, X2, U, o, o; ndrange=(npl, nc1, nt))

    Z = KernelAbstractions.zeros(backend, T, ndf, nc1, nt)
    nn(Z, batch.Rx, Qx, o, o; ndrange=(ndf, nc1, nt))
    nn(Z, batch.Ry, Qy, o, o; ndrange=(ndf, nc1, nt))
    nn(Z, batch.Ru, U, o, o; ndrange=(ndf, nc1, nt))

    ae = similar(Z, ndf, ndf, nt)
    ae .= .-(batch.Alam .+ view(Z, :, 1:ndf, :))
    fe = Z[:, nc1, :]
    KernelAbstractions.synchronize(backend)

    return (; ae, fe, U, Qx, Qy)
end

"""
    hdg_recover(batch, loc, x)

Recovers `uh (npl, nt)` and `qh (npl, 2, nt)` from the global trace vector `x`
(on the same backend as `batch`) using the local solution maps from
[`hdg_local_solves`](@ref) — a batched matvec, no per-element re-solve.
"""
function hdg_recover(batch::HDGBatch{T}, loc, x) where {T}
    backend = KernelAbstractions.get_backend(batch.D)
    npl, ndf, nt = batch.npl, batch.ndf, batch.nt

    m = KernelAbstractions.zeros(backend, T, ndf, nt)
    _gather_trace!(backend)(m, x, batch.elcon; ndrange=(ndf, nt))

    uh = KernelAbstractions.zeros(backend, T, npl, nt)
    qx = KernelAbstractions.zeros(backend, T, npl, nt)
    qy = KernelAbstractions.zeros(backend, T, npl, nt)
    _recover!(backend)(uh, qx, qy, loc.U, loc.Qx, loc.Qy, m; ndrange=(npl, nt))
    KernelAbstractions.synchronize(backend)

    return uh, qx, qy
end

"""
    hdg_parsolve_batched(master, mesh, source, dbc, param;
                         ArrayT=Array, T=Float64, kwargs...)

Fully device-resident counterpart of [`hdg_parsolve`](@ref): batched element
assembly ([`hdg_local_solves`](@ref)) and local recovery
([`hdg_recover`](@ref)) run on the backend of `ArrayT` alongside the
[`hdg_gmres_ka`](@ref) trace solve. Only the Dirichlet-BC application and the
face-block global assembly (`hdg_densesystem`) remain on the CPU. `kwargs` are
forwarded to `hdg_gmres_ka`.

Returns `(uh, qh, uhath, niter)` with the same shapes as `hdg_parsolve`
(`uh (npl, 1, nt)`, `qh (npl, 2, nt)`, `uhath (nps, nf)`).
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

    uh_d, qx_d, qy_d = hdg_recover(batch, loc, x)
    uh = Array{Float64}(undef, batch.npl, 1, batch.nt)
    uh[:, 1, :] .= Array(uh_d)
    qh = Array{Float64}(undef, batch.npl, 2, batch.nt)
    qh[:, 1, :] .= Array(qx_d)
    qh[:, 2, :] .= Array(qy_d)
    uhath = reshape(Float64.(Array(x)), mesh.porder + 1, :)

    return uh, qh, uhath, stats.niter
end
