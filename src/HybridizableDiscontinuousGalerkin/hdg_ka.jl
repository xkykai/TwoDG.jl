# KernelAbstractions + Krylov.jl iterative solver for the HDG trace system
# (Phase 3 of GPU_PLAN.md). Element assembly and local recovery stay on the
# CPU (element-sequential, done once per solve); the GMRES iteration — the
# face-parallel matvec and block-Jacobi preconditioner that dominate the
# runtime — runs on any KA backend. Move the system to a GPU with
# `Adapt.adapt(CuArray, sys)` or by passing `ArrayT=CuArray` to
# `hdg_parsolve_ka`.

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using Adapt
using Krylov
using LinearAlgebra

"""
    HDGSystem(ae, fe, mesh; T=Float64)

The statically condensed HDG trace system in face-block format, ready for the
KA/Krylov iterative solver. All fields are plain arrays, so the whole struct
moves to a GPU with `Adapt.adapt(CuArray, sys)`.

Fields (`ncf` trace DOFs per face, `nbf = 2nfe - 1` neighbor blocks, `nf` faces):
- `A (ncf, ncf, nbf, nf)`: global matrix; block `k` of face `i` couples face
  `i` to face `f2f[i, k]` (block 1 is the self/diagonal block).
- `B (ncf, ncf, nf)`: inverted diagonal blocks (block-Jacobi preconditioner).
- `b (ncf * nf)`: right-hand side.
- `f2f (nf, nbf)`: face-to-face connectivity, `0` = no neighbor.
"""
struct HDGSystem{T, V <: AbstractVector{T}, A3 <: AbstractArray{T, 3},
                 A4 <: AbstractArray{T, 4}, I2 <: AbstractMatrix{Int32}}
    A   :: A4
    B   :: A3
    b   :: V
    f2f :: I2
end

Adapt.@adapt_structure HDGSystem

Base.eltype(::HDGSystem{T}) where {T} = T

function HDGSystem(ae, fe, mesh; T::Type{<:AbstractFloat}=Float64)
    nps = mesh.porder + 1
    A, b = hdg_densesystem(ae, fe, mesh.f, mesh.t2f, nps)
    B = compute_blockjacobi(A)
    f2f = mesh.f2f === nothing ? mkf2f(mesh.f, mesh.t2f) : mesh.f2f
    return HDGSystem(T.(A), T.(B), T.(b), Int32.(f2f))
end

# One workitem per (row, face): y[i, fc] = Σ_k A[i, :, k, fc] ⋅ x[:, f2f[fc, k]]
@kernel function _hdg_matvec!(y, @Const(A), @Const(x), @Const(f2f))
    i, fc = @index(Global, NTuple)
    T = eltype(y)
    ncf = size(A, 1)
    nbf = size(f2f, 2)

    acc = zero(T)
    @inbounds for k in 1:nbf
        j = f2f[fc, k]
        if j > 0
            for m in 1:ncf
                acc += A[i, m, k, fc] * x[m, j]
            end
        end
    end
    @inbounds y[i, fc] = acc
end

# One workitem per (row, face): y[i, fc] = B[i, :, fc] ⋅ x[:, fc]
@kernel function _hdg_blockjacobi!(y, @Const(B), @Const(x))
    i, fc = @index(Global, NTuple)
    T = eltype(y)
    ncf = size(B, 1)

    acc = zero(T)
    @inbounds for m in 1:ncf
        acc += B[i, m, fc] * x[m, fc]
    end
    @inbounds y[i, fc] = acc
end

# Minimal linear-operator wrappers so Krylov.jl can drive the KA kernels
# (Krylov only needs eltype, size, and 3-arg mul!).
abstract type HDGOperator{T} end

Base.eltype(::HDGOperator{T}) where {T} = T
Base.size(op::HDGOperator) = (op.n, op.n)
Base.size(op::HDGOperator, i::Integer) = i <= 2 ? op.n : 1

struct HDGMatVecOp{T, A4 <: AbstractArray{T, 4}, I2} <: HDGOperator{T}
    A   :: A4
    f2f :: I2
    ncf :: Int
    nf  :: Int
    n   :: Int
end

HDGMatVecOp(sys::HDGSystem) =
    HDGMatVecOp(sys.A, sys.f2f, size(sys.A, 1), size(sys.A, 4), length(sys.b))

function LinearAlgebra.mul!(y::AbstractVector, op::HDGMatVecOp, x::AbstractVector)
    backend = KernelAbstractions.get_backend(op.A)
    _hdg_matvec!(backend)(reshape(y, op.ncf, op.nf), op.A,
                          reshape(x, op.ncf, op.nf), op.f2f;
                          ndrange=(op.ncf, op.nf))
    KernelAbstractions.synchronize(backend)
    return y
end

struct HDGBlockJacobiOp{T, A3 <: AbstractArray{T, 3}} <: HDGOperator{T}
    B   :: A3
    ncf :: Int
    nf  :: Int
    n   :: Int
end

HDGBlockJacobiOp(sys::HDGSystem) =
    HDGBlockJacobiOp(sys.B, size(sys.B, 1), size(sys.B, 3), length(sys.b))

function LinearAlgebra.mul!(y::AbstractVector, op::HDGBlockJacobiOp, x::AbstractVector)
    backend = KernelAbstractions.get_backend(op.B)
    _hdg_blockjacobi!(backend)(reshape(y, op.ncf, op.nf), op.B,
                               reshape(x, op.ncf, op.nf);
                               ndrange=(op.ncf, op.nf))
    KernelAbstractions.synchronize(backend)
    return y
end

"""
    hdg_gmres_ka(sys::HDGSystem; restart=80, tol=1e-6, maxit=2000,
                 preconditioner=true, verbose=false)

Solves the HDG trace system with restarted GMRES (Krylov.jl), left-
preconditioned by the block-Jacobi blocks of `sys`, on whatever backend `sys`
lives on. Returns `(x, stats)` with `stats::Krylov.SimpleStats`.
"""
function hdg_gmres_ka(sys::HDGSystem{T}; restart=80, tol=1e-6, maxit=2000,
                      preconditioner=true, verbose=false) where {T}
    Aop = HDGMatVecOp(sys)
    M = preconditioner ? HDGBlockJacobiOp(sys) : I
    x, stats = Krylov.gmres(Aop, sys.b; M, ldiv=false, restart=true,
                            memory=restart, atol=zero(T), rtol=T(tol),
                            itmax=maxit, verbose=verbose ? 1 : 0)
    return x, stats
end

"""
    hdg_parsolve_ka(master, mesh, source, dbc, param;
                    ArrayT=Array, T=Float64, kwargs...)

GPU-capable counterpart of [`hdg_parsolve`](@ref): identical CPU element
assembly and local recovery, but the trace system is solved with
[`hdg_gmres_ka`](@ref) on the backend of `ArrayT` (e.g. `ArrayT=CuArray` with
CUDA.jl loaded). `kwargs` are forwarded to `hdg_gmres_ka`.

Returns `(uh, qh, uhath, niter)` like `hdg_parsolve`.
"""
function hdg_parsolve_ka(master, mesh, source, dbc, param;
                         ArrayT=Array, T::Type{<:AbstractFloat}=Float64, kwargs...)
    ae, fe = hdg_elemmats(master, mesh, source, dbc, param)
    sys = adapt(ArrayT, HDGSystem(ae, fe, mesh; T))
    x, stats = hdg_gmres_ka(sys; kwargs...)
    uhath = Array(x)
    uh, qh = hdg_localrecovery(master, mesh, Float64.(uhath), source, param)
    return uh, qh, uhath, stats.niter
end
