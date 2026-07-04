# Batched CG element system + matrix-free Krylov solver on any KA backend
# (Phase 4 of GPU_PLAN.md). The element integrals of `elemmat_cg` restated as
# per-element dense algebra over the Geometry module's tables:
#
#   ae = κ (Sx Dxᵀ + Sy Dyᵀ) − (c₁ Sx + c₂ Sy) Φᵀ + s M,   fe = Φ (w .* f(pg))
#
# with Sx/Sy the quadrature+Jacobian-weighted derivative tables, Dx/Dy their
# unweighted counterparts, Φ the shape values and M the element mass matrix.
# Dirichlet conditions are imposed by *symmetric* elimination (rows and
# columns), which for the homogeneous data supported here gives the same
# solution as the legacy row-zeroing but keeps the operator SPD when there is
# no convection — so the pure-diffusion case can use conjugate gradients
# (and, on the direct path, Cholesky).
#
# The global solve is matrix-free: one fused gather → batched-matvec →
# atomic-scatter kernel per operator application (`CGMatVecOp`), Jacobi
# (diagonal) preconditioning, Krylov.jl `cg`/`gmres`. Everything the solve
# touches is a plain array, so `ArrayT=CuArray` runs the iteration on a GPU.

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using Atomix
using Adapt
using Krylov
using LinearAlgebra
using SparseArrays
using TwoDG.Geometry: RefTables, element_geometry!

"""
    cg_element_system(mesh, master, source, param; T=Float64, eliminate=true)

Batched CG element matrices and load vectors: `ae (npl, npl, nt)`,
`fe (npl, nt)`, the global Dirichlet-node mask `dirichlet (nn)`, and whether
the (eliminated) operator is symmetric (`c == 0`).

Geometry is evaluated isoparametrically on `mesh.pcg[mesh.tcg[e, :], :]` —
the same coordinates `elemmat_cg` uses (they can differ from `dgnodes` by
`cgmesh`'s deduplication rounding), so parity with `elemmat_cg` is exact.
`eliminate=false` skips the symmetric Dirichlet elimination (used by the
parity tests).
"""
function cg_element_system(mesh, master, source, param;
                           T::Type{<:AbstractFloat}=Float64, eliminate::Bool=true)
    κ, c, s = param.κ, param.c, param.s
    npl = size(mesh.plocal, 1)
    nt = size(mesh.tcg, 1)
    rt = RefTables(master)
    ng = length(rt.gwgh)

    ae = zeros(T, npl, npl, nt)
    fe = zeros(T, npl, nt)

    Tg = promote_type(eltype(mesh.pcg), eltype(rt.gwgh))   # geometry precision
    Threads.@threads for e in 1:nt
        Sx = Matrix{Tg}(undef, npl, ng)
        Sy = Matrix{Tg}(undef, npl, ng)
        wjac = Vector{Tg}(undef, ng)
        pg = Matrix{Tg}(undef, ng, 2)
        coords = mesh.pcg[mesh.tcg[e, :], :]

        # always the isoparametric (per-quad-point) path, matching elemmat_cg
        M = element_geometry!(Sx, Sy, wjac, pg, rt, coords; verts=nothing)

        Dx = Sx ./ wjac'                       # unweighted physical derivatives
        Dy = Sy ./ wjac'
        A = κ .* (Sx * Dx' .+ Sy * Dy')        # diffusion
        A .-= (c[1] .* Sx .+ c[2] .* Sy) * rt.shap'   # convection
        A .+= s .* M                           # reaction
        ae[:, :, e] .= A

        fq = [wjac[g] * source(pg[g, 1], pg[g, 2]) for g in 1:ng]
        fe[:, e] .= rt.shap * fq
    end

    dirichlet = cg_dirichlet_mask(mesh, master)

    if eliminate
        for e in 1:nt, i in 1:npl
            if dirichlet[mesh.tcg[e, i]]
                ae[i, :, e] .= zero(T)
                ae[:, i, e] .= zero(T)
                fe[i, e] = zero(T)
            end
        end
    end

    symmetric = iszero(c[1]) && iszero(c[2])
    return ae, fe, dirichlet, symmetric
end

"Global mask of CG nodes on the (Dirichlet) boundary, from the face list."
function cg_dirichlet_mask(mesh, master)
    dirichlet = falses(size(mesh.pcg, 1))
    for i in findall(<(0), mesh.f[:, 4])
        el = mesh.f[i, 3]
        ipl = sum(mesh.t[el, :]) - sum(mesh.f[i, 1:2])
        isl = findfirst(==(ipl), mesh.t[el, :])
        dirichlet[mesh.tcg[el, master.perm[:, isl, 1]]] .= true
    end
    return dirichlet
end

"""
    cg_assemble(ae, fe, tcg, dirichlet) -> (K, F)

Triplet-based global assembly (`sparse(I, J, V)` sums duplicates): the
sparse matrix `K` with unit diagonal on Dirichlet rows and the load vector
`F`. Replaces the legacy incremental `K[i, j] += ...` insertion, which
re-shuffled the CSC structure per entry.
"""
function cg_assemble(ae::AbstractArray{T, 3}, fe, tcg, dirichlet) where {T}
    npl = size(ae, 1)
    nt = size(ae, 3)
    nn = length(dirichlet)
    ndir = count(dirichlet)

    Iv = Vector{Int}(undef, npl * npl * nt + ndir)
    Jv = Vector{Int}(undef, npl * npl * nt + ndir)
    Vv = Vector{T}(undef, npl * npl * nt + ndir)
    idx = 1
    for e in 1:nt, j in 1:npl, i in 1:npl
        Iv[idx] = tcg[e, i]
        Jv[idx] = tcg[e, j]
        Vv[idx] = ae[i, j, e]
        idx += 1
    end
    for d in findall(dirichlet)
        Iv[idx] = d; Jv[idx] = d; Vv[idx] = one(T)
        idx += 1
    end
    K = sparse(Iv, Jv, Vv, nn, nn)

    F = zeros(T, nn)
    for e in 1:nt, i in 1:npl
        F[tcg[e, i]] += fe[i, e]
    end
    return K, F
end

# ------------------------------------------------- matrix-free KA operator

# One workitem per (local node, element): row i of element e's matrix times
# the gathered element vector, atomically scattered to the global row.
# Dirichlet rows receive no contribution (their ae rows/columns are zero);
# the identity on them is applied by broadcast in mul!.
@kernel function _cg_matvec!(y, @Const(ae), @Const(x), @Const(tcg))
    i, e = @index(Global, NTuple)
    T = eltype(y)
    npl = size(ae, 1)
    acc = zero(T)
    @inbounds for j in 1:npl
        acc += ae[i, j, e] * x[tcg[e, j]]
    end
    @inbounds Atomix.@atomic y[tcg[e, i]] += acc
end

# One workitem per (local node, element): accumulate the global diagonal.
@kernel function _cg_diag!(d, @Const(ae), @Const(tcg))
    i, e = @index(Global, NTuple)
    @inbounds Atomix.@atomic d[tcg[e, i]] += ae[i, i, e]
end

"""
    CGMatVecOp(ae, tcg, dirichlet)

Matrix-free CG stiffness operator for Krylov.jl (implements `eltype`, `size`,
3-arg `mul!`): `y = K x` as gather → batched row-matvec → atomic scatter,
with the identity on Dirichlet rows. Lives on whatever backend its arrays
live on.
"""
struct CGMatVecOp{T, A3 <: AbstractArray{T, 3}, I2 <: AbstractMatrix{Int32}, BV}
    ae        :: A3
    tcg       :: I2
    dirichlet :: BV
    npl       :: Int
    nt        :: Int
    n         :: Int
end

CGMatVecOp(ae, tcg, dirichlet) =
    CGMatVecOp(ae, tcg, dirichlet, size(ae, 1), size(ae, 3), length(dirichlet))

Base.eltype(::CGMatVecOp{T}) where {T} = T
Base.size(op::CGMatVecOp) = (op.n, op.n)
Base.size(op::CGMatVecOp, i::Integer) = i <= 2 ? op.n : 1

function LinearAlgebra.mul!(y::AbstractVector, op::CGMatVecOp, x::AbstractVector)
    backend = KernelAbstractions.get_backend(op.ae)
    fill!(y, zero(eltype(y)))
    _cg_matvec!(backend)(y, op.ae, x, op.tcg; ndrange=(op.npl, op.nt))
    KernelAbstractions.synchronize(backend)
    y .= ifelse.(op.dirichlet, x, y)
    return y
end

"Jacobi (diagonal) preconditioner: `y = x ./ d`."
struct CGJacobiOp{T, V <: AbstractVector{T}}
    d :: V
end

Base.eltype(::CGJacobiOp{T}) where {T} = T
Base.size(op::CGJacobiOp) = (length(op.d), length(op.d))
Base.size(op::CGJacobiOp, i::Integer) = i <= 2 ? length(op.d) : 1

function LinearAlgebra.mul!(y::AbstractVector, op::CGJacobiOp, x::AbstractVector)
    y .= x ./ op.d
    return y
end

# --------------------------------------------------------------- driver

"""
    cg_parsolve(mesh, master, source, param; T=Float64, ArrayT=Array,
                tol=1e-10, maxit=5000, restart=80, preconditioner=true,
                verbose=false) -> (uh, energy, niter)

Matrix-free iterative CG solve on any KA backend: conjugate gradients when
the operator is symmetric (no convection), restarted GMRES otherwise, both
Jacobi-preconditioned. `ArrayT=CuArray` (with CUDA.jl loaded) runs the whole
iteration on the GPU. Returns the same `uh (npl, nt)` / `energy` as
[`cg_solve`](@ref), plus the Krylov iteration count.
"""
function cg_parsolve(mesh, master, source, param;
                     T::Type{<:AbstractFloat}=Float64, ArrayT=Array,
                     tol=1e-10, maxit=5000, restart=80, preconditioner=true,
                     verbose=false)
    ae, fe, dirichlet, symmetric = cg_element_system(mesh, master, source, param; T)
    nn = length(dirichlet)
    npl, nt = size(fe, 1), size(fe, 2)

    F = zeros(T, nn)
    for e in 1:nt, i in 1:npl
        F[mesh.tcg[e, i]] += fe[i, e]
    end

    ae_d = adapt(ArrayT, ae)
    tcg_d = adapt(ArrayT, Int32.(mesh.tcg))
    dir_d = adapt(ArrayT, collect(dirichlet))
    b = adapt(ArrayT, F)

    Aop = CGMatVecOp(ae_d, tcg_d, dir_d)

    backend = KernelAbstractions.get_backend(ae_d)
    d = KernelAbstractions.zeros(backend, T, nn)
    _cg_diag!(backend)(d, ae_d, tcg_d; ndrange=(npl, nt))
    KernelAbstractions.synchronize(backend)
    d .= ifelse.(dir_d, one(T), d)
    M = preconditioner ? CGJacobiOp(d) : I

    x, stats = if symmetric
        Krylov.cg(Aop, b; M, ldiv=false, atol=zero(T), rtol=T(tol),
                  itmax=maxit, verbose=verbose ? 1 : 0)
    else
        Krylov.gmres(Aop, b; M, ldiv=false, restart=true, memory=restart,
                     atol=zero(T), rtol=T(tol), itmax=maxit,
                     verbose=verbose ? 1 : 0)
    end

    Ax = similar(x)
    mul!(Ax, Aop, x)
    energy = 0.5 * dot(x, Ax) - dot(x, b)

    xh = Array(x)
    uh = Matrix{T}(undef, npl, nt)
    for e in 1:nt, i in 1:npl
        uh[i, e] = xh[mesh.tcg[e, i]]
    end
    return uh, energy, stats.niter
end
