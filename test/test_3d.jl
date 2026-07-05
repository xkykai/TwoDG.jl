# 3D mesh, connectivity, and orientation (THREED_PLAN Phase C).
#
# The two-tet orientation battery is the single most important test of the 3D
# plan: a shared triangular face can be seen in 6 relative orientations, and a
# subtly wrong permutation table corrupts fluxes/traces *silently* (the
# residual stays finite, the rates quietly degrade). Every assertion here runs
# over all vertex orderings of the two-tet mesh.

using TwoDG
using Test
using StaticArrays
using LinearAlgebra

# Golden regression value (see the "golden regression" testset below).
const GOLDEN_EULER_BOX3D = 0.83769243146855621

"3D mesh invariants: Euler face counting, adjacency, t2o consistency, volumes."
function check_tet_mesh_invariants(mesh)
    (; p, t, f, t2f, t2o) = mesh
    nf, nt = size(f, 1), size(t, 1)
    bnd = f[:, 5] .< 0
    ni = count(!, bnd)

    # interior faces first, boundary faces last; handshake: 4 faces/element
    @test !any(bnd[1:ni]) && all(bnd[(ni + 1):end])
    @test 4nt == 2ni + (nf - ni)

    # each face's vertices belong to its adjacent element(s)
    @test all(issubset(f[i, 1:3], t[f[i, 4], :]) for i in 1:nf)
    @test all(issubset(f[i, 1:3], t[f[i, 5], :]) for i in 1:ni)

    # t2f/f agree; reference counts; stored traversal = left element's outward
    # traversal (o = 1); the right element of a conforming, positively
    # oriented mesh sees the face reflected (o ∈ 4:6)
    refcount = zeros(Int, nf)
    ok_adj = true
    ok_orient = true
    for el in 1:nt, k in 1:4
        fc = t2f[el, k]
        refcount[fc] += 1
        ok_adj &= (f[fc, 4] == el || f[fc, 5] == el)
        fv = face_vertices(Val(3), k)
        mine = ntuple(j -> t[el, fv[j]], 3)
        stored = (f[fc, 1], f[fc, 2], f[fc, 3])
        o = face_orientation(stored, mine, Val(3))
        ok_orient &= t2o[el, k] == o
        f[fc, 4] == el && (ok_orient &= o == 1)
        f[fc, 5] == el && (ok_orient &= o in 4:6)
    end
    @test ok_adj && ok_orient
    @test all(refcount[1:ni] .== 2) && all(refcount[(ni + 1):end] .== 1)

    # positively oriented elements
    vol6(el) = det(hcat((p[t[el, v], :] - p[t[el, 1], :] for v in 2:4)...))
    @test all(vol6(el) > 0 for el in 1:nt)
end

# all positively oriented vertex orderings of a tet (the 12 even permutations)
function positive_orderings(verts, p)
    out = NTuple{4, Int}[]
    for a in 1:4, b in 1:4, c in 1:4, d in 1:4
        length(unique((a, b, c, d))) == 4 || continue
        tt = (verts[a], verts[b], verts[c], verts[d])
        v6 = det(hcat((p[tt[v], :] - p[tt[1], :] for v in 2:4)...))
        v6 > 0 && push!(out, tt)
    end
    return out
end

@testset "3D meshes and orientation" begin
    @testset "two-tet orientation battery (p = $porder)" for porder in (1, 3)
        pts = [0.0 0.0 0.0;    # 1: apex of tet A
               1.0 0.0 0.0;    # 2 ┐
               0.0 1.0 0.0;    # 3 │ shared face
               0.0 0.0 1.0;    # 4 ┘
               1.0 1.0 1.0]    # 5: apex of tet B
        bnds = (all=p -> trues(size(p, 1)),)

        orderingsA = positive_orderings((1, 2, 3, 4), pts)
        orderingsB = positive_orderings((5, 2, 3, 4), pts)
        seen_codes = Set{Int}()

        # a modest sweep of tet A's orderings × ALL of tet B's — every
        # relative orientation of the shared face arises
        for ta in orderingsA[1:3:end], tb in orderingsB
            t = [ta[1] ta[2] ta[3] ta[4]; tb[1] tb[2] tb[3] tb[4]]
            mesh = discretize(MeshGeometry(copy(pts), t; boundaries=bnds), porder)
            check_tet_mesh_invariants(mesh)

            master = ReferenceElement(mesh)
            ctx = DGContext(master, mesh)

            # (i) the shared face's nodes coincide row-by-row through facecon:
            # both sides list their volume nodes in the face's canonical order
            el, er = ctx.f_el[1, 1], ctx.f_el[1, 2]
            xl = mesh.dgnodes[ctx.facecon[:, 1, 1], :, el]
            xr = mesh.dgnodes[ctx.facecon[:, 2, 1], :, er]
            @test maximum(abs, xl - xr) < 1e-13

            # (ii) free-stream preservation: a constant state has zero
            # residual to machine precision (volume/face metrics + orientation
            # must all be consistent for the telescoping cancellation)
            eq = ConvectionEquation([0.7, -0.4, 1.1])
            phys = DGPhysics(eq; boundary_conditions=(Dirichlet(2.5),))
            u = fill(2.5, ctx.npl, 1, ctx.nt)
            r = rinvexpl_ka(ctx, phys, u, 0.0)
            @test maximum(abs, r) < 1e-10

            # (iii) HDG trace continuity: the two elements' elcon entries for
            # the shared face reference the same global trace dof at the same
            # physical point
            sl = findfirst(==(1), mesh.t2f[el, :])
            sr = findfirst(==(1), mesh.t2f[er, :])
            nps = size(mesh.elcon, 1)
            xg = Dict{Int, Vector{Float64}}()
            okt = true
            for k in 1:nps
                g = mesh.elcon[k, sl, el]
                xg[g] = mesh.dgnodes[master.perm[k, sl, 1], :, el]
            end
            for k in 1:nps
                g = mesh.elcon[k, sr, er]
                okt &= haskey(xg, g) &&
                       maximum(abs, xg[g] - mesh.dgnodes[master.perm[k, sr, 1], :, er]) < 1e-13
            end
            @test okt

            push!(seen_codes, mesh.t2o[er, sr])
        end

        # the battery genuinely exercised every reflected orientation
        @test seen_codes == Set([4, 5, 6])
    end

    @testset "box mesh (Kuhn 6-tet) invariants" begin
        mesh = mkmesh_box(3, 3, 3, 2)
        check_tet_mesh_invariants(mesh)
        @test ndims(mesh) == 3
        @test size(mesh.t, 1) == 6 * 2 * 2 * 2
        @test boundary_names(mesh) == [:left, :right, :front, :back, :bottom, :top]

        # total volume = 1, total boundary area = 6 through the quadrature
        master = ReferenceElement(mesh)
        ctx = DGContext(master, mesh)
        @test abs(sum(ctx.wjac) - 1.0) < 1e-12
        bnd = (ctx.ni + 1):ctx.nf
        @test abs(sum(ctx.dws[:, bnd]) - 6.0) < 1e-12

        # every boundary normal points outward from the box center
        ok = true
        for fc in bnd, g in 1:ctx.ngf
            x = ctx.pfg[g, :, fc] .- 0.5
            ok &= sum(ctx.nlg[g, :, fc] .* x) > 0
        end
        @test ok

        # free-stream preservation on the box
        eq = ConvectionEquation([1.0, 0.5, -0.25])
        phys = DGPhysics(eq; boundary_conditions=ntuple(_ -> Dirichlet(1.5), 6))
        u = fill(1.5, ctx.npl, 1, ctx.nt)
        r = rinvexpl_ka(ctx, phys, u, 0.0)
        @test maximum(abs, r) < 1e-10
    end

    @testset "tet uniref (Bey) + fixmesh" begin
        p0, t0 = make_box_mesh(2, 2, 2)
        p1, t1 = uniref(p0, t0, 2)
        @test size(t1, 1) == 64 * size(t0, 1)

        # children tile the box: positive volumes summing to 1
        vols = TwoDG.Meshes.simpvol(p1, t1)
        @test all(vols .> 0)
        @test abs(sum(vols) - 1.0) < 1e-12

        # the refined mesh is conforming — the full connectivity invariants
        # hold (Bey children of neighboring tets must share matching faces)
        bnds = (all=p -> trues(size(p, 1)),)
        mesh = discretize(MeshGeometry(p1, t1; boundaries=bnds), 2)
        check_tet_mesh_invariants(mesh)

        # fixmesh: merges duplicated vertices and repairs orientation
        pd = vcat(p0, p0[1:1, :])                       # duplicate of vertex 1
        td = copy(t0)
        td[1, 3], td[1, 4] = td[1, 4], td[1, 3]         # break orientation
        td[2, findfirst(==(1), td[2, :])] = size(p0, 1) + 1  # use the duplicate
        pf, tf = fixmesh(pd, td)
        @test size(pf, 1) == size(p0, 1)
        @test all(TwoDG.Meshes.simpvol(pf, tf) .> 0)
    end

    @testset "3D convection: exact translation, O(h^{p+1}) convergence" begin
        # u_t + ∇·(v u) = 0 with constant v transports the profile unchanged:
        # u(x, t) = u0(x - vt). A globally smooth sine product with the exact
        # solution as time-dependent Dirichlet inflow data, so every mesh in
        # the sequence resolves it and no boundary tail pollutes the rate.
        # The horizon must be long enough for the wave to cross O(1) cells:
        # at short times the error is truncation-dominated and measures only
        # O(h^p) — a rate-testing pitfall, not a discretization property.
        v = SVector(1.0, 0.5, 0.25)
        u0(x) = sin(π * x[1]) * sin(π * x[2]) * sin(π * x[3])
        uexact(x, t) = u0(x .- v .* t)
        dt, nstep = 1e-3, 200
        tfinal = dt * nstep

        porder = 2
        errs = Float64[]
        for nx in (3, 5, 9)
            mesh = mkmesh_box(nx, nx, nx, porder)
            master = ReferenceElement(mesh)
            ctx = DGContext(master, mesh)
            phys = DGPhysics(ConvectionEquation(collect(v));
                             boundary_conditions=ntuple(_ -> Dirichlet(uexact), 6))
            u = reshape(mapslices(x -> u0(SVector{3}(x)), mesh.dgnodes, dims=2),
                        ctx.npl, 1, ctx.nt)
            rk4_ka!(ctx, phys, u, 0.0, dt, nstep)
            uex = reshape(mapslices(x -> uexact(SVector{3}(x), tfinal), mesh.dgnodes, dims=2),
                          ctx.npl, 1, ctx.nt)
            err = sqrt(sum(ctx.wjac .* ((ctx.shap' * (u[:, 1, :] .- uex[:, 1, :])) .^ 2)))
            push!(errs, err)
        end
        rates = log2.(errs[1:end-1] ./ errs[2:end])
        @test rates[end] > porder + 0.6   # design rate p+1 (h halves 5 -> 9; measured 2.84)
    end

    @testset "curved 3D boundary: sphere octant (C4 projection, D2 free stream)" begin
        # Unit corner tet refined twice; the hypotenuse-plane vertices are
        # projected radially so the linear mesh respects the unit sphere, and
        # discretize projects the high-order boundary-face nodes (THREED_PLAN
        # C4). Run twice: the sphere as one curved boundary, and split into
        # two curved boundaries sharing rim edges — the shared nodes then
        # carry two curved tags and go through the alternating-projection
        # (edge-before-face) path, which must stay conforming.
        p0 = [0.0 0 0; 1.0 0 0; 0 1.0 0; 0 0 1.0]
        t0 = reshape([1, 2, 3, 4], 1, 4)
        p1, t1 = uniref(p0, t0, 2)
        for i in axes(p1, 1)
            if abs(sum(p1[i, :]) - 1) < 1e-12
                p1[i, :] ./= norm(p1[i, :])
            end
        end

        ϵ = 1e-6
        onsphere = p -> vec(sqrt.(sum(abs2, p; dims=2)) .> 0.8)
        fdsphere = x -> sqrt(x[1]^2 + x[2]^2 + x[3]^2) - 1
        planes = (xy=p -> p[:, 3] .< ϵ, xz=p -> p[:, 2] .< ϵ, yz=p -> p[:, 1] .< ϵ)
        planefds = (x -> x[3], x -> x[2], x -> x[1])

        configs = (
            (boundaries=(sphere=onsphere, planes...),
             curved=[:sphere], fd=(fdsphere, planefds...), nsph=1),
            (boundaries=(cap=p -> onsphere(p) .& (p[:, 3] .> 0.4),
                         band=p -> onsphere(p) .& (p[:, 3] .<= 0.4), planes...),
             curved=[:cap, :band], fd=(fdsphere, fdsphere, planefds...), nsph=2),
        )
        for cfg in configs
            geo = MeshGeometry(p1, t1; boundaries=cfg.boundaries,
                               curved=cfg.curved, fd=cfg.fd)
            mesh = discretize(geo, 3)
            check_tet_mesh_invariants(mesh)

            master = ReferenceElement(mesh)
            ctx = DGContext(master, mesh)

            # every curved-face node landed on the sphere
            ok = true
            for i in axes(mesh.f, 1)
                -mesh.f[i, 5] in 1:cfg.nsph || continue
                el = mesh.f[i, 4]
                s = findfirst(==(i), mesh.t2f[el, :])
                for k in axes(master.perm, 1)
                    x = mesh.dgnodes[master.perm[k, s, 1], :, el]
                    ok &= abs(norm(x) - 1) < 1e-10
                end
            end
            @test ok

            # conforming: interior-face node point sets coincide side to side
            okc = true
            for fc in 1:ctx.ni
                el, er = ctx.f_el[fc, 1], ctx.f_el[fc, 2]
                xl = mesh.dgnodes[ctx.facecon[:, 1, fc], :, el]
                xr = mesh.dgnodes[ctx.facecon[:, 2, fc], :, er]
                okc &= maximum(abs, xl - xr) < 1e-12
            end
            @test okc

            # isoparametric metric: valid elements, octant volume π/6 and
            # sphere-patch area π/2 to the geometric approximation order
            # (measured 2.1e-4 / 4.2e-4 at p = 3 on the 64-tet octant)
            @test all(>(0), ctx.wjac)
            @test abs(sum(ctx.wjac) - π / 6) < 5e-4
            sph = [fc for fc in (ctx.ni + 1):ctx.nf if -mesh.f[fc, 5] in 1:cfg.nsph]
            @test abs(sum(ctx.dws[:, sph]) - π / 2) < 1e-3

            # free-stream preservation on the curved mesh
            nbnd = 3 + cfg.nsph
            phys = DGPhysics(ConvectionEquation([0.8, -0.3, 0.55]);
                             boundary_conditions=ntuple(_ -> Dirichlet(1.2), nbnd))
            u = fill(1.2, ctx.npl, 1, ctx.nt)
            r = rinvexpl_ka(ctx, phys, u, 0.0)
            @test maximum(abs, r) < 5e-10
        end
    end

    @testset "3D heat equation (LDG): exact decay, O(h^{p+1}) convergence" begin
        # u_t = κ Δu on the unit box with homogeneous Dirichlet boundaries:
        # u = sin(πx) sin(πy) sin(πz) exp(-3π²κt) is exact.
        κ = 0.05
        u0f(x, y, z) = sin(π * x) * sin(π * y) * sin(π * z)
        dt, nstep = 2e-4, 500
        tfinal = dt * nstep
        exact(x, y, z) = u0f(x, y, z) * exp(-3π^2 * κ * tfinal)

        porder = 2
        errs = map((3, 5)) do n
            mesh = mkmesh_box(n, n, n, porder)
            master = ReferenceElement(mesh)
            phys = DGPhysics(ConvectionDiffusionEquation(SVector(0.0, 0.0, 0.0), κ);
                             boundary_conditions=ntuple(_ -> Dirichlet(0.0), 6),
                             stabilization=LDGStabilization(10.0, 0.0))
            u = initu(mesh, 1, [u0f])
            ctx = DGContext(master, mesh)
            rk4_ka!(rldgexpl!, ctx, phys, u, 0.0, dt, nstep)
            l2error(mesh, u[:, 1, :], exact)
        end
        @test log2(errs[1] / errs[2]) > porder + 0.5
    end

    @testset "3D Float32 contexts (Euler + LDG viscous)" begin
        γ = 1.4
        mesh = mkmesh_box(3, 3, 3, 2)
        master = ReferenceElement(mesh)
        relerr(a, b) = norm(a .- b) / norm(b)

        # Euler: smooth, strictly physical state
        ρ(x, y, z) = 1.0 + 0.1 * exp(-30 * ((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2))
        ics = [ρ,
               (x, y, z) -> 0.3 * ρ(x, y, z),
               (x, y, z) -> 0.05 * ρ(x, y, z),
               (x, y, z) -> -0.1 * ρ(x, y, z),
               (x, y, z) -> 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2 + 0.1^2) * ρ(x, y, z)]
        uinf = [1.0, 0.3, 0.05, -0.1, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2 + 0.1^2)]
        phys = DGPhysics(EulerEquations{3}(γ=γ);
                         boundary_conditions=ntuple(_ -> FarField(uinf), 6))
        u = initu(mesh, 5, ics)
        r64 = rinvexpl_ka(DGContext(master, mesh), phys, u, 0.0)
        ctx32 = DGContext(master, mesh; T=Float32)
        @test eltype(ctx32) == Float32
        r32 = rinvexpl_ka(ctx32, phys, Float32.(u), 0.0f0)
        @test eltype(r32) == Float32
        @test all(isfinite, r32)
        @test relerr(Float64.(r32), r64) < 1e-3

        # LDG viscous path
        eq = ConvectionDiffusionEquation(SVector(0.4, -0.3, 0.2), 0.01)
        phys_v = DGPhysics(eq; boundary_conditions=ntuple(_ -> Dirichlet(0.0), 6),
                           stabilization=LDGStabilization(10.0, 0.5))
        uv = initu(mesh, 1, [(x, y, z) -> exp(-4 * ((x - 0.5)^2 + y^2 + z^2))])
        rv64 = rldgexpl_ka(DGContext(master, mesh), phys_v, uv, 0.0)
        rv32 = rldgexpl_ka(ctx32, phys_v, Float32.(uv), 0.0f0)
        @test eltype(rv32) == Float32
        @test all(isfinite, rv32)
        @test relerr(Float64.(rv32), rv64) < 1e-3
    end

    @testset "3D Euler isentropic vortex: O(h^{p+1}) convergence + golden" begin
        # Isentropic vortex (axis along z) superposed on a uniform free stream
        # and advected with it — an exact 3D Euler solution (the classic 2D
        # vortex is z-invariant; run *as 3D* with w∞ ≠ 0 so all three flux
        # directions and all six boundaries are exercised). Spatial rescaling
        # by the core radius R preserves exactness. Solved through the
        # user-facing DGProblem/solve path with dt from compute_dt.
        γ = 1.4
        β, R = 1.0, 0.25
        v∞ = SVector(1.0, 0.5, 0.25)
        function primitives(x, y, t)
            x̂ = (x - 0.5 - v∞[1] * t) / R
            ŷ = (y - 0.5 - v∞[2] * t) / R
            f = exp((1 - x̂^2 - ŷ^2) / 2)
            Θ = 1 - (γ - 1) * β^2 / (8γ * π^2) * f^2       # temperature
            ρ = Θ^(1 / (γ - 1))
            vel = SVector(v∞[1] - β / (2π) * ŷ * f, v∞[2] + β / (2π) * x̂ * f, v∞[3])
            return ρ, vel, Θ^(γ / (γ - 1))
        end
        function conserved(x, y, z, t)
            ρ, vel, p = primitives(x, y, t)
            return SVector(ρ, ρ * vel[1], ρ * vel[2], ρ * vel[3],
                           p / (γ - 1) + ρ * sum(abs2, vel) / 2)
        end
        uex(x, t) = conserved(x[1], x[2], x[3], t)

        porder = 2
        tfinal = 0.1
        eq = EulerEquations{3}(γ=γ)
        errs = map((3, 5, 9)) do nx
            mesh = mkmesh_box(nx, nx, nx, porder)
            prob = DGProblem(eq, mesh;
                             bc=ntuple(_ -> Dirichlet(uex), 6),
                             u0=[(x, y, z) -> conserved(x, y, z, 0.0)[c] for c in 1:5])
            dt = compute_dt(prob; cfl=0.3)
            @test 0 < dt < 0.1
            nstep = ceil(Int, tfinal / dt)
            sol = solve(prob, RK4(); dt=tfinal / nstep, tfinal)
            l2error(mesh, sol.u[:, 1, :], (x, y, z) -> conserved(x, y, z, tfinal)[1])
        end
        rates = log2.(errs[1:end-1] ./ errs[2:end])
        @test rates[end] > porder + 0.6   # design rate p+1 (h halves 5 -> 9)
    end

    @testset "golden regression: 3D Euler residual" begin
        # Pinned 2026-07-05 from the CPU Float64 run of the dimension-generic
        # flux/Roe implementation, cross-validated by the vortex convergence
        # test above and the 2D golden (which the generalization left intact).
        γ = 1.4
        mesh = mkmesh_box(3, 3, 3, 3)
        master = ReferenceElement(mesh)
        ρ(x, y, z) = 1.0 + 0.1 * exp(-30 * ((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2))
        ics = [ρ,
               (x, y, z) -> 0.3 * ρ(x, y, z),
               (x, y, z) -> 0.05 * ρ(x, y, z),
               (x, y, z) -> -0.1 * ρ(x, y, z),
               (x, y, z) -> 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2 + 0.1^2) * ρ(x, y, z)]
        uinf = [1.0, 0.3, 0.05, -0.1, 1.0 / (γ - 1) + 0.5 * (0.3^2 + 0.05^2 + 0.1^2)]
        phys = DGPhysics(EulerEquations{3}(γ=γ);
                         boundary_conditions=ntuple(_ -> FarField(uinf), 6))
        u = initu(mesh, 5, ics)
        r = rinvexpl_ka(DGContext(master, mesh), phys, u, 0.0)
        @test norm(r) ≈ GOLDEN_EULER_BOX3D rtol = 1e-9
    end
end

@testset "HDG 3D Poisson (THREED_PLAN Phase E1)" begin
    # -Δu = 3π² sin(πx)sin(πy)sin(πz) with homogeneous Dirichlet data: the
    # source is evaluated at quadrature points and the boundary data is exact
    # under nodal interpolation, the two data rules p+2 superconvergence
    # requires. Batched assembly runs on the 6-orientation elcon path.
    exact(x, y, z) = sin(π * x) * sin(π * y) * sin(π * z)
    source(p) = reshape(3π^2 .* sin.(π .* p[:, 1]) .* sin.(π .* p[:, 2]) .*
                        sin.(π .* p[:, 3]), :, 1)
    dbc(p) = zeros(size(p, 1), 1)
    param = Dict(:kappa => 1.0, :c => [0.0, 0.0, 0.0], :taud => 1.0)

    porder = 2
    ngauss = 4 * (porder + 1)

    @testset "O(h^{p+1}) in u, O(h^{p+2}) postprocessing" begin
        errs_u, errs_ustar = Float64[], Float64[]
        for n in (3, 5)   # h = 1/2, 1/4
            mesh = mkmesh_box(n, n, n, porder)
            master = ReferenceElement(mesh, ngauss)
            mesh1 = mkmesh_box(n, n, n, porder + 1)
            master1 = ReferenceElement(mesh1, ngauss)

            u, q, _ = hdg_direct_batched(master, mesh, source, dbc, param)
            ustar = hdg_postprocess(master, mesh, master1, mesh1, u,
                                    q ./ param[:kappa])
            push!(errs_u, l2error(mesh, u[:, 1, :], exact))
            push!(errs_ustar, l2error(mesh1, ustar[:, 1, :], exact))
        end
        # measured 2.79 / 3.89 at these resolutions (design 3 / 4)
        @test log2(errs_u[1] / errs_u[2]) > porder + 0.5
        @test log2(errs_ustar[1] / errs_ustar[2]) > porder + 1.5
        @test all(errs_ustar .< errs_u ./ 4)
    end

    @testset "GMRES trace solve matches direct" begin
        # exercises hdg_densesystem's orientation gather (all 6 codes) and the
        # block-Jacobi preconditioner on tetrahedral face blocks
        mesh = mkmesh_box(3, 3, 3, porder)
        master = ReferenceElement(mesh, ngauss)
        ud, qd, _ = hdg_direct_batched(master, mesh, source, dbc, param)
        up, qp, _, niter = hdg_parsolve_batched(master, mesh, source, dbc, param;
                                                tol=1e-12, restart=200)
        @test niter > 0
        @test norm(vec(up) .- vec(ud)) / norm(vec(ud)) < 1e-9
        @test norm(vec(qp) .- vec(qd)) / norm(vec(qd)) < 1e-9
    end

    @testset "Interface: HDGProblem + PoissonEquation{3}" begin
        # nonzero (harmonic) Dirichlet data exercises the (x, y, z) lowering
        exact_a(x, y, z) = exact(x, y, z) + 0.5x - 0.25y + 0.125z
        prob = HDGProblem(PoissonEquation{3}(), mkmesh_box(4, 4, 4, porder);
                          bc=Dirichlet(exact_a), source=source)
        sol = solve(prob, Direct())
        @test size(sol.q, 2) == 3
        @test l2error(sol, exact_a) < 0.02
    end
end

@testset "HDG 3D convection-diffusion (THREED_PLAN Phase E2)" begin
    # c·∇u - κΔu = f with a manufactured solution and homogeneous Dirichlet
    # data; the nonzero velocity exercises the convective trace terms and the
    # tau = taud + |c·n| stabilization on all 6 face orientations.
    κ = 1.0
    c = [1.0, 0.5, 0.25]
    u_ex(x, y, z) = sin(π * x) * sin(π * y) * sin(π * z)
    function source(p)
        f = similar(p, size(p, 1))
        for i in axes(p, 1)
            x, y, z = p[i, 1], p[i, 2], p[i, 3]
            sx, sy, sz = sin(π * x), sin(π * y), sin(π * z)
            cx, cy, cz = cos(π * x), cos(π * y), cos(π * z)
            f[i] = 3π^2 * κ * sx * sy * sz +
                   π * (c[1] * cx * sy * sz + c[2] * sx * cy * sz +
                        c[3] * sx * sy * cz)
        end
        return reshape(f, :, 1)
    end
    dbc(p) = zeros(size(p, 1), 1)
    param = Dict(:kappa => κ, :c => c, :taud => 1.0)

    porder = 2
    errs = map((3, 5)) do n
        mesh = mkmesh_box(n, n, n, porder)
        master = ReferenceElement(mesh, 4 * (porder + 1))
        u, _, _ = hdg_direct_batched(master, mesh, source, dbc, param)
        l2error(mesh, u[:, 1, :], u_ex)
    end
    @test log2(errs[1] / errs[2]) > porder + 0.5   # measured 2.80 (design 3)

    # Interface lowering: Dim inferred from the SVector velocity, GMRES path
    prob = HDGProblem(ConvectionDiffusionEquation(SVector(1.0, 0.5, 0.25), κ),
                      mkmesh_box(4, 4, 4, porder); bc=Dirichlet(0.0), source=source)
    sol = solve(prob)
    @test sol.iterations > 0
    @test l2error(sol, u_ex) < 0.01
end
