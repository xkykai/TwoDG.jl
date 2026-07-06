# ------------------------------------------------------------------------------
# KA kernels (state-varying assembly + recovery), dimension-generic: `Dim` is
# read off the array shapes (fixed small trip counts) or passed as `Val(DIM)`
# where a compile-time value keeps the inner loops unrolled.
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

# X[m, n, e] = Σ_g A[m, g, e] * ug[g, ci, e] * shap[n, g]  (A = shapd slice d)
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

# A = A0 + Newton convection blocks + dtinv·M on the velocity diagonal blocks:
# velocity block (bi, bj) gets -X[·,·,bj,bi] (the δu_bj term of ∂_j(u_i u_j)),
# the diagonal blocks additionally -Ku = -Σ_d X[·,·,d,d] and dtinv·M; pressure
# rows/cols (incl. the gauge row) are pure A0.
@kernel function _ns_assemble_A!(A, @Const(A0), @Const(X), @Const(M), dtinv, npl)
    i, j, e = @index(Global, NTuple)
    Dim = size(X, 3)
    @inbounds begin
        base = A0[i, j, e]
        bi = (i - 1) ÷ npl + 1
        bj = (j - 1) ÷ npl + 1
        if bi <= Dim && bj <= Dim
            ii = i - (bi - 1) * npl
            jj = j - (bj - 1) * npl
            base -= X[ii, jj, bj, bi, e]
            if bi == bj
                for d in 1:Dim
                    base -= X[ii, jj, d, d, e]
                end
                base += dtinv * M[ii, jj, e]
            end
        end
        A[i, j, e] = base
    end
end

# r column of Bhat: r_c += -(Ku u)_c + M (fsrc_c + dtinv uold_c) + rext_c,
# using (Ku u)_c[i] = Σ_g (Σ_d shapd[i,g,d] ug_d) ug_c.
@kernel function _ns_rhs!(Bhat, @Const(shapd), @Const(ug), @Const(M),
                          @Const(fsrc), @Const(uold), @Const(rext), dtinv,
                          npl, ncB)
    i, c, e = @index(Global, NTuple)
    T = eltype(Bhat)
    ng = size(ug, 1)
    Dim = size(ug, 2)
    acc = zero(T)
    @inbounds for g in 1:ng
        du = zero(T)
        for d in 1:Dim
            du += shapd[i, g, d, e] * ug[g, d, e]
        end
        acc -= du * ug[g, c, e]
    end
    @inbounds for j in 1:npl
        acc += M[i, j, e] * (fsrc[j, c, e] + dtinv * uold[j, c, e])
    end
    @inbounds acc += rext[i, c, e]
    @inbounds Bhat[(c - 1) * npl + i, ncB, e] += acc
end

# Newton linearization of the trace convection (λ·n)λ_i, accumulated into the
# local RHS block Bhat, the trace-test block Hlam and the flux RHS rH. One
# workitem per element (the serial face loop sidesteps races on nodes shared
# by several faces).
@kernel function _ns_faces!(Bhat, Hlam, rH, @Const(lam), @Const(shf),
                            @Const(wds), @Const(fn), @Const(perm),
                            npl, nps, ncB, ::Val{DIM}) where {DIM}
    e = @index(Global)
    T = eltype(Bhat)
    nfe = DIM + 1
    nfc = nfe * nps
    nq = size(shf, 2)
    @inbounds for s in 1:nfe
        for g in 1:nq
            λ = ntuple(Val(DIM)) do c
                acc = zero(T)
                for a in 1:nps
                    acc += shf[a, g] * lam[(c - 1) * nfc + (s - 1) * nps + a, e]
                end
                acc
            end
            λn = zero(T)
            for d in 1:DIM
                λn += λ[d] * fn[g, d, s, e]
            end
            w = wds[g, s, e]
            for a in 1:nps
                sha = shf[a, g]
                p = perm[a, s]
                cola = (s - 1) * nps + a
                rc = w * sha * λn
                for i in 1:DIM
                    Bhat[(i - 1) * npl + p, ncB, e] += rc * λ[i]
                    rH[(i - 1) * nfc + cola, e] += rc * λ[i]
                end
                for b in 1:nps
                    colb = (s - 1) * nps + b
                    wab = w * sha * shf[b, g]
                    for i in 1:DIM, j in 1:DIM
                        cij = wab * (fn[g, j, s, e] * λ[i] + (i == j ? λn : zero(T)))
                        Bhat[(i - 1) * npl + p, (j - 1) * nfc + colb, e] += cij
                        Hlam[(i - 1) * nfc + cola, (j - 1) * nfc + colb, e] += cij
                    end
                end
            end
        end
    end
end

# gather the interleaved global trace into element-local [λ₁; …; λ_Dim] blocks
@kernel function _ns_gather!(lam, @Const(Λ), @Const(elcon), nps, Dim)
    k, e = @index(Global, NTuple)
    nfc = size(elcon, 2) * nps
    c = (k - 1) ÷ nfc + 1
    ℓ = k - (c - 1) * nfc
    s = (ℓ - 1) ÷ nps + 1
    a = ℓ - (s - 1) * nps
    @inbounds g = elcon[a, s, e]
    @inbounds lam[k, e] = Λ[Dim * (g - 1) + c]
end

# (u, p) recovery x = Z r-col − Z_B λ − Z_bρ ρ, routed into un/pn
@kernel function _ns_recover!(un, pn, @Const(Z), @Const(lam), @Const(ρ), npl, ncB)
    i, e = @index(Global, NTuple)
    Dim = size(un, 2)
    nlam = ncB - 2
    @inbounds acc = Z[i, ncB, e] - Z[i, ncB - 1, e] * ρ[e]
    @inbounds for k in 1:nlam
        acc -= Z[i, k, e] * lam[k, e]
    end
    b = (i - 1) ÷ npl + 1
    @inbounds if b <= Dim
        un[i - (b - 1) * npl, b, e] = acc
    else
        pn[i - Dim * npl, e] = acc
    end
end

# velocity gradient L_ij = M⁻¹(E_j λ_i − C_jᵀ u_i), column (i-1)·Dim + j
@kernel function _ns_gradient!(Ln, @Const(MiE), @Const(MiC), @Const(lam),
                               @Const(un), nfc)
    i, e = @index(Global, NTuple)
    T = eltype(Ln)
    npl = size(un, 1)
    Dim = size(un, 2)
    @inbounds for ci in 1:Dim, d in 1:Dim
        acc = zero(T)
        for k in 1:nfc
            acc += MiE[i, k, d, e] * lam[(ci - 1) * nfc + k, e]
        end
        for j in 1:npl
            acc -= MiC[i, j, d, e] * un[j, ci, e]
        end
        Ln[i, (ci - 1) * Dim + d, e] = acc
    end
end

# scalar-trace gather (face-scalar DOFs, no interleaving)
@kernel function _cd_gather!(th, @Const(Θ), @Const(elcon), nps)
    k, e = @index(Global, NTuple)
    s = (k - 1) ÷ nps + 1
    a = k - (s - 1) * nps
    @inbounds th[k, e] = Θ[elcon[a, s, e]]
end

# trace convection ⟨(λ·n) θ̂, ·⟩ into the local RHS block B and trace block Hlam
@kernel function _cd_faces!(B, Hlam, @Const(lam), @Const(shf), @Const(wds),
                            @Const(fn), @Const(perm), nps, ncB,
                            ::Val{DIM}) where {DIM}
    e = @index(Global)
    T = eltype(B)
    nfe = DIM + 1
    nfc = nfe * nps
    nq = size(shf, 2)
    @inbounds for s in 1:nfe
        for g in 1:nq
            λn = zero(T)
            for c in 1:DIM
                acc = zero(T)
                for a in 1:nps
                    acc += shf[a, g] * lam[(c - 1) * nfc + (s - 1) * nps + a, e]
                end
                λn += acc * fn[g, c, s, e]
            end
            w = wds[g, s, e]
            for a in 1:nps
                sha = shf[a, g]
                p = perm[a, s]
                cola = (s - 1) * nps + a
                for b in 1:nps
                    colb = (s - 1) * nps + b
                    c = w * sha * shf[b, g] * λn
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

# scalar gradient q_d = M⁻¹(E_d θ̂ − C_dᵀ θ)
@kernel function _cd_gradq!(qn, @Const(MiE), @Const(MiC), @Const(th), @Const(θn))
    i, e = @index(Global, NTuple)
    T = eltype(qn)
    nfc = size(MiE, 2)
    npl = size(θn, 1)
    Dim = size(qn, 2)
    @inbounds for d in 1:Dim
        acc = zero(T)
        for k in 1:nfc
            acc += MiE[i, k, d, e] * th[k, e]
        end
        for j in 1:npl
            acc -= MiC[i, j, d, e] * θn[j, e]
        end
        qn[i, d, e] = acc
    end
end
