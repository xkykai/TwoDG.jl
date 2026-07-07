#=
Natural convection in a differentially heated cubic cavity, solved in 3D with
the GPU-accelerated batched HDG Navier-Stokes + scalar-transport solvers
(`hdg_ns_step_batched` / `hdg_cd_step_batched`): batched element assembly,
local solves and recovery run on the GPU through KernelAbstractions; the
condensed trace systems are CPU sparse LU with pattern and numeric
factorization reused across steps.

Nondimensionalization by the buoyancy velocity U = sqrt(gβΔT L):
ν = sqrt(Pr/Ra), κ = 1/sqrt(Ra·Pr). Hot wall θ = +1/2 at x = 0, cold wall
θ = -1/2 at x = 1, the four remaining walls insulated, no-slip everywhere,
gravity in -z (buoyancy source θ in the w-momentum equation). The transient
from the conductive state forms boundary layers and a rising hot-wall plume
that stratifies the core — rendered as temperature isosurfaces.

Usage:
    JULIA_LOAD_PATH='@;@cuda;@v#.#;@stdlib' julia +1.12 --project=. \
        examples/hdg3d/runhdg3d_boussinesq_animation.jl [solve|render|all]

`solve` writes one Serialization snapshot per frame to
./output/boussinesq3d_anim/ (restart-friendly); `render` (needs GLMakie on
the load path) turns whatever snapshots exist into the GIF. The GPU is used
when CUDA.jl is available in the environment stack.
=#
using TwoDG
using LinearAlgebra
using Printf
using Serialization

#%% Parameters
Ra = 1e5
Pr = 0.71
ν = sqrt(Pr / Ra)
κ = 1 / sqrt(Ra * Pr)

# overridable for smoke runs: TWODG_BOUSS3D_N=5 TWODG_BOUSS3D_TEND=0.1 ...
n = parse(Int, get(ENV, "TWODG_BOUSS3D_N", "9"))       # (n-1)³ cubes × 6 tets
porder = 2
Δt = parse(Float64, get(ENV, "TWODG_BOUSS3D_DT", "0.05"))
t_end = parse(Float64, get(ENV, "TWODG_BOUSS3D_TEND", "15.0"))
save_every = 4                 # steps between animation frames
τ = 1.0

outdir = joinpath(@__DIR__, "..", "..", "output", "boussinesq3d_anim")
gifpath = joinpath(@__DIR__, "..", "..", "figures", "hdg_ns_boussinesq3d_ra1e5.gif")

mode = isempty(ARGS) ? "all" : ARGS[1]

mesh = mkmesh_box(n, n, n, porder)
master = ReferenceElement(mesh, 3 * (porder + 1))
npl, nt = size(mesh.dgnodes, 1), size(mesh.t, 1)
nps, nf = size(mesh.elcon, 1) ÷ 4, size(mesh.f, 1)

# box tags: 1 left (x=0, hot), 2 right (x=1, cold), 3/4 front/back, 5/6 bottom/top
dbc(p) = [0.0, 0.0, 0.0]
tbc(p, tag) = tag == 1 ? (:d, 0.5) : tag == 2 ? (:d, -0.5) : (:n, 0.0)

"Hot-wall Nusselt number from the temperature gradient q = ∇θ."
function nusselt_hot(mesh, master, q)
    fe = master.face
    fshap = fe.shap[:, 1, :]
    rt = TwoDG.Geometry.RefTables(master)
    ngf = length(rt.gwf)
    nlg = zeros(ngf, 3); dws = zeros(ngf); pfg = zeros(ngf, 3)
    Nu = 0.0
    for i in axes(mesh.f, 1)
        mesh.f[i, 5] == -1 || continue
        it = mesh.f[i, 4]
        lf = findfirst(==(i), mesh.t2f[it, :])
        pp = master.perm[:, lf, mesh.t2o[it, lf]]
        TwoDG.Geometry.face_geometry!(nlg, dws, pfg, rt, mesh.dgnodes[pp, :, it])
        Nu -= sum(dws .* (fshap' * q[pp, 1, it]))
    end
    return Nu
end

#%% Solve phase
if mode in ("solve", "all")
    ArrayT = try
        @eval using CUDA
        @assert CUDA.functional()
        @info "running on GPU: $(CUDA.name(CUDA.device()))"
        CUDA.CuArray
    catch
        @warn "CUDA not available - running the batched path on CPU threads"
        Array
    end

    mkpath(outdir)
    nsteps = round(Int, t_end / Δt)
    dtinv = 1 / Δt
    statepath = joinpath(outdir, "state.bin")
    logio = open(joinpath(outdir, "progress.log"), "a")

    if isfile(statepath)
        st = deserialize(statepath)
        θ, u, Λ = st.θ, st.u, st.Λ
        t, step0, iframe = st.t, st.step, st.iframe
        hist_t, hist_nu, hist_ke = st.hist_t, st.hist_nu, st.hist_ke
        println(logio, "resuming from step $step0, t = $t"); flush(logio)
    else
        θ = [0.5 - mesh.dgnodes[k, 1, it] for k in 1:npl, it in 1:nt]
        u = zeros(npl, 3, nt)
        Λ = zeros(3 * nps * nf)
        t = 0.0
        step0 = 0
        iframe = 0
        hist_t, hist_nu, hist_ke = Float64[], Float64[], Float64[]
        serialize(joinpath(outdir, "frame_0000.bin"), (t=0.0, θ=θ))
    end
    nscache = nothing
    cdcache = nothing

    wall0 = time()
    for step in step0+1:nsteps
        global θ, u, Λ, nscache, cdcache, t, iframe

        θres = hdg_cd_step_batched(master, mesh, κ, tbc; τ, u, Λ, θold=θ,
                                   dtinv, ArrayT, cache=cdcache)
        θ = θres.θ
        cdcache = θres.cache

        src = zeros(npl, 3, nt)
        src[:, 3, :] .= θ                     # buoyancy in +z for hot fluid
        uold = copy(u)
        res = hdg_ns_step_batched(master, mesh, ν, dbc; τ, source=src,
                                  u, Λ, uold, dtinv, ArrayT, cache=nscache)
        u, Λ = res.u, res.Λ
        nscache = res.cache

        t += Δt
        Nu = nusselt_hot(mesh, master, θres.q)
        ke = sum(u .^ 2)
        push!(hist_t, t); push!(hist_nu, Nu); push!(hist_ke, ke)
        all(isfinite, u) || error("solution diverged at t = $t")

        if step % save_every == 0
            iframe += 1
            serialize(joinpath(outdir, @sprintf("frame_%04d.bin", iframe)), (t=t, θ=θ))
        end
        if step % 5 == 0 || step == nsteps
            serialize(statepath, (; θ, u, Λ, t, step, iframe, hist_t, hist_nu, hist_ke))
        end
        if step % 10 == 0 || step == 1
            println(logio, @sprintf("step %4d/%d  t = %5.2f  Nu = %7.3f  KE = %.3e  (%.1f s/step)",
                                    step, nsteps, t, Nu, ke, (time() - wall0) / (step - step0)))
            flush(logio)
        end
    end
    serialize(joinpath(outdir, "history.bin"), (t=hist_t, nu=hist_nu, ke=hist_ke))
    close(logio)
end

#%% Render phase: resample θ onto a regular grid, isosurfaces via GLMakie
if mode in ("render", "all")
    using GLMakie
end

if mode in ("render", "all")
    # --- static point location: bucket elements by the structured cells ---
    ngrid = 49
    xs = range(0.005, 0.995, length=ngrid)
    grid = [(x, y, z) for x in xs, y in xs, z in xs]

    ncell = n - 1
    cellof(x) = clamp(floor(Int, x * ncell) + 1, 1, ncell)
    buckets = [Int[] for _ in 1:ncell, _ in 1:ncell, _ in 1:ncell]
    for e in 1:nt
        c = vec(sum(mesh.p[mesh.t[e, :], :]; dims=1)) ./ 4
        push!(buckets[cellof(c[1]), cellof(c[2]), cellof(c[3])], e)
    end

    # for each grid point: containing tet + interpolation weights (npl,)
    "barycentric coordinates of x in tet e (affine)"
    function bary(mesh, e, x)
        v = mesh.p[mesh.t[e, :], :]
        A = vcat(v', ones(1, 4))
        return A \ [x[1], x[2], x[3], 1.0]
    end
    locs = Vector{Tuple{Int, Vector{Float64}}}(undef, length(grid))
    plref = mesh.plocal[:, 2:4]
    for (ig, x) in enumerate(vec(grid))
        e_found = 0
        λbest = nothing
        for e in buckets[cellof(x[1]), cellof(x[2]), cellof(x[3])]
            λ = bary(mesh, e, collect(x))
            if minimum(λ) > -1e-9
                e_found = e
                λbest = λ
                break
            end
        end
        e_found == 0 && error("grid point $x not located")
        ξ = λbest[2:4]
        sh = TwoDG.Masters.shape3d(porder, mesh.plocal, reshape(ξ, 1, 3))[:, 1, 1]
        locs[ig] = (e_found, sh)
    end

    sample(θ) = reshape([dot(sh, @view θ[:, e]) for (e, sh) in locs],
                        ngrid, ngrid, ngrid)

    frames = sort(filter(f -> startswith(f, "frame_"), readdir(outdir)))
    @info "rendering $(length(frames)) frames"
    θs = [deserialize(joinpath(outdir, f)) for f in frames]

    # function-form `lift` (not the @lift macro): the macro would fail to
    # expand when this file is loaded in `solve` mode without GLMakie
    iobs = Observable(1)
    vol = lift(i -> Float32.(sample(θs[i].θ)), iobs)
    title_str = lift(i -> @sprintf("Heated cavity, Ra = 10⁵ — 3D HDG Navier-Stokes on the GPU   t = %.1f", θs[i].t), iobs)

    fig = Figure(size=(720, 620), backgroundcolor=:white)
    ax = Axis3(fig[1, 1]; aspect=(1, 1, 1), title=title_str,
               xlabel="x", ylabel="y", zlabel="z", azimuth=-0.65π, elevation=0.15π)
    contour!(ax, (xs[1], xs[end]), (xs[1], xs[end]), (xs[1], xs[end]), vol;
             levels=[-0.35, -0.2, -0.05, 0.05, 0.2, 0.35],
             colormap=Reverse(:RdBu), colorrange=(-0.5, 0.5), alpha=0.4,
             transparency=true)
    Colorbar(fig[1, 2], colormap=Reverse(:RdBu), limits=(-0.5, 0.5), label="θ")

    mkpath(dirname(gifpath))
    mp4path = replace(gifpath, ".gif" => ".mp4")
    record(fig, mp4path, eachindex(θs); framerate=12) do i
        i % 20 == 0 && @info "frame $i / $(length(θs))"
        iobs[] = i
        ax.azimuth[] = -0.65π + 0.3π * (i - 1) / max(length(θs) - 1, 1)
    end
    @info "wrote $mp4path ($(round(filesize(mp4path) / 1e6, digits=1)) MB)"

    # reduced palette + no dither keeps the GIF small (see the 2D example)
    ffmpeg = Sys.which("ffmpeg") === nothing ?
        GLMakie.Makie.FFMPEG_jll.ffmpeg() : `ffmpeg`
    vf = "fps=12,scale=640:-1:flags=lanczos,split[s0][s1];" *
         "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=none"
    run(`$ffmpeg -y -loglevel error -i $mp4path -vf $vf $gifpath`)
    @info "wrote $gifpath ($(round(filesize(gifpath) / 1e6, digits=1)) MB)"
end
