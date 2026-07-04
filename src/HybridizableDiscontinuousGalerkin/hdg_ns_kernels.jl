# ------------------------------------------------------------------------------
# KA kernels (state-varying assembly + recovery)
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

# X[m, n, e] = Σ_g A[m, g, e] * ug[g, ci, e] * shap[n, g]  (A = shapx or shapy)
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

# A = A0 + Newton convection blocks + dtinv·M on the velocity diagonal blocks.
# Block (1,1) gets -X1 - Ku = -2X1 - Y2, block (2,2) gets -Y2 - Ku = -X1 - 2Y2,
# off-diagonal velocity blocks -Y1 / -X2; pressure rows/cols (incl. the gauge
# row) are pure A0.
@kernel function _ns_assemble_A!(A, @Const(A0), @Const(X1), @Const(X2),
                                 @Const(Y1), @Const(Y2), @Const(M), dtinv, npl)
    i, j, e = @index(Global, NTuple)
    @inbounds begin
        base = A0[i, j, e]
        if i <= npl && j <= npl
            base += -2 * X1[i, j, e] - Y2[i, j, e] + dtinv * M[i, j, e]
        elseif i <= npl && j <= 2 * npl
            base += -Y1[i, j - npl, e]
        elseif i <= 2 * npl && j <= npl
            base += -X2[i - npl, j, e]
        elseif i <= 2 * npl && j <= 2 * npl
            base += -X1[i - npl, j - npl, e] - 2 * Y2[i - npl, j - npl, e] +
                    dtinv * M[i - npl, j - npl, e]
        end
        A[i, j, e] = base
    end
end

# r column of Bhat: r_c += -(Ku u)_c + M (fsrc_c + dtinv uold_c) + rext_c,
# using (Ku u)_c[i] = Σ_g (shapx[i,g] ug1 + shapy[i,g] ug2) ug_c.
@kernel function _ns_rhs!(Bhat, @Const(shapx), @Const(shapy), @Const(ug),
                          @Const(M), @Const(fsrc), @Const(uold), @Const(rext),
                          dtinv, npl, ncB)
    i, c, e = @index(Global, NTuple)
    T = eltype(Bhat)
    ng = size(ug, 1)
    acc = zero(T)
    @inbounds for g in 1:ng
        acc -= (shapx[i, g, e] * ug[g, 1, e] + shapy[i, g, e] * ug[g, 2, e]) * ug[g, c, e]
    end
    @inbounds for j in 1:npl
        acc += M[i, j, e] * (fsrc[j, c, e] + dtinv * uold[j, c, e])
    end
    @inbounds acc += rext[i, c, e]
    @inbounds Bhat[(c - 1) * npl + i, ncB, e] += acc
end

# Newton linearization of the trace convection (λ·n)λ_i, accumulated into the
# local RHS block Bhat, the trace-test block Hlam and the flux RHS rH. One
# workitem per element (the serial 3-edge loop sidesteps corner-node races).
@kernel function _ns_faces!(Bhat, Hlam, rH, @Const(lam), @Const(sh1d),
                            @Const(wds), @Const(fn1), @Const(fn2),
                            @Const(perm), npl, nps, ncB)
    e = @index(Global)
    T = eltype(Bhat)
    nfc = 3 * nps
    nq = size(sh1d, 2)
    @inbounds for s in 1:3
        for g in 1:nq
            λ1 = zero(T)
            λ2 = zero(T)
            for a in 1:nps
                col = (s - 1) * nps + a
                sh = sh1d[a, g]
                λ1 += sh * lam[col, e]
                λ2 += sh * lam[nfc + col, e]
            end
            n1 = fn1[g, s, e]
            n2 = fn2[g, s, e]
            w = wds[g, s, e]
            λn = λ1 * n1 + λ2 * n2
            for a in 1:nps
                sha = sh1d[a, g]
                p = perm[a, s]
                cola = (s - 1) * nps + a
                rc = w * sha * λn
                Bhat[p, ncB, e] += rc * λ1
                Bhat[npl + p, ncB, e] += rc * λ2
                rH[cola, e] += rc * λ1
                rH[nfc + cola, e] += rc * λ2
                for b in 1:nps
                    colb = (s - 1) * nps + b
                    wab = w * sha * sh1d[b, g]
                    c11 = wab * (n1 * λ1 + λn)
                    c12 = wab * n2 * λ1
                    c21 = wab * n1 * λ2
                    c22 = wab * (n2 * λ2 + λn)
                    Bhat[p, colb, e] += c11
                    Bhat[p, nfc + colb, e] += c12
                    Bhat[npl + p, colb, e] += c21
                    Bhat[npl + p, nfc + colb, e] += c22
                    Hlam[cola, colb, e] += c11
                    Hlam[cola, nfc + colb, e] += c12
                    Hlam[nfc + cola, colb, e] += c21
                    Hlam[nfc + cola, nfc + colb, e] += c22
                end
            end
        end
    end
end

# gather the interleaved global trace into element-local [λ1; λ2] blocks
@kernel function _ns_gather!(lam, @Const(Λ), @Const(elcon), nps)
    k, e = @index(Global, NTuple)
    nfc = 3 * nps
    c = k <= nfc ? 1 : 2
    ℓ = k - (c - 1) * nfc
    s = div(ℓ - 1, nps) + 1
    a = ℓ - (s - 1) * nps
    @inbounds g = elcon[a, s, e]
    @inbounds lam[k, e] = Λ[2 * (g - 1) + c]
end

# (u, p) recovery x = Z r-col − Z_B λ − Z_bρ ρ, routed into un/pn
@kernel function _ns_recover!(un, pn, @Const(Z), @Const(lam), @Const(ρ), npl, ncB)
    i, e = @index(Global, NTuple)
    T = eltype(un)
    nlam = ncB - 2
    @inbounds acc = Z[i, ncB, e] - Z[i, ncB - 1, e] * ρ[e]
    @inbounds for k in 1:nlam
        acc -= Z[i, k, e] * lam[k, e]
    end
    @inbounds if i <= npl
        un[i, 1, e] = acc
    elseif i <= 2 * npl
        un[i - npl, 2, e] = acc
    else
        pn[i - 2 * npl, e] = acc
    end
end

# velocity gradient L_ij = M⁻¹(E_j λ_i − C_jᵀ u_i), columns L11 L12 L21 L22
@kernel function _ns_gradient!(Ln, @Const(MiE1), @Const(MiE2), @Const(MiCx),
                               @Const(MiCy), @Const(lam), @Const(un), nfc)
    i, e = @index(Global, NTuple)
    T = eltype(Ln)
    npl = size(un, 1)
    l11 = zero(T); l12 = zero(T); l21 = zero(T); l22 = zero(T)
    @inbounds for k in 1:nfc
        λ1k = lam[k, e]
        λ2k = lam[nfc + k, e]
        l11 += MiE1[i, k, e] * λ1k
        l12 += MiE2[i, k, e] * λ1k
        l21 += MiE1[i, k, e] * λ2k
        l22 += MiE2[i, k, e] * λ2k
    end
    @inbounds for j in 1:npl
        u1j = un[j, 1, e]
        u2j = un[j, 2, e]
        l11 -= MiCx[i, j, e] * u1j
        l12 -= MiCy[i, j, e] * u1j
        l21 -= MiCx[i, j, e] * u2j
        l22 -= MiCy[i, j, e] * u2j
    end
    @inbounds begin
        Ln[i, 1, e] = l11
        Ln[i, 2, e] = l12
        Ln[i, 3, e] = l21
        Ln[i, 4, e] = l22
    end
end

# scalar-trace gather (face-scalar DOFs, no interleaving)
@kernel function _cd_gather!(th, @Const(Θ), @Const(elcon), nps)
    k, e = @index(Global, NTuple)
    s = div(k - 1, nps) + 1
    a = k - (s - 1) * nps
    @inbounds th[k, e] = Θ[elcon[a, s, e]]
end

# trace convection ⟨(λ·n) θ̂, ·⟩ into the local RHS block B and trace block Hlam
@kernel function _cd_faces!(B, Hlam, @Const(lam), @Const(sh1d), @Const(wds),
                            @Const(fn1), @Const(fn2), @Const(perm), nps, ncB)
    e = @index(Global)
    T = eltype(B)
    nfc = 3 * nps
    nq = size(sh1d, 2)
    @inbounds for s in 1:3
        for g in 1:nq
            λ1 = zero(T)
            λ2 = zero(T)
            for a in 1:nps
                col = (s - 1) * nps + a
                sh = sh1d[a, g]
                λ1 += sh * lam[col, e]
                λ2 += sh * lam[nfc + col, e]
            end
            λn = λ1 * fn1[g, s, e] + λ2 * fn2[g, s, e]
            w = wds[g, s, e]
            for a in 1:nps
                sha = sh1d[a, g]
                p = perm[a, s]
                cola = (s - 1) * nps + a
                for b in 1:nps
                    colb = (s - 1) * nps + b
                    c = w * sha * sh1d[b, g] * λn
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

# scalar gradient q_j = M⁻¹(E_j θ̂ − C_jᵀ θ)
@kernel function _cd_gradq!(qn, @Const(MiE1), @Const(MiE2), @Const(MiCx),
                            @Const(MiCy), @Const(th), @Const(θn))
    i, e = @index(Global, NTuple)
    T = eltype(qn)
    nfc = size(MiE1, 2)
    npl = size(θn, 1)
    q1 = zero(T); q2 = zero(T)
    @inbounds for k in 1:nfc
        thk = th[k, e]
        q1 += MiE1[i, k, e] * thk
        q2 += MiE2[i, k, e] * thk
    end
    @inbounds for j in 1:npl
        θj = θn[j, e]
        q1 -= MiCx[i, j, e] * θj
        q2 -= MiCy[i, j, e] * θj
    end
    @inbounds qn[i, 1, e] = q1
    @inbounds qn[i, 2, e] = q2
end

