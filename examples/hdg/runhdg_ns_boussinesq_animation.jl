#=
Natural convection in a differentially heated square cavity at high Rayleigh
number, run with the GPU-accelerated batched HDG Navier-Stokes solver
(`hdg_ns_step_batched` / `hdg_cd_step_batched`, GPU_PLAN.md Phase 5): the
batched element assembly, local solves and recovery execute on the GPU
through KernelAbstractions; the condensed trace systems are CPU sparse LU
with the sparsity pattern and numeric factorization reused across steps.

Same physics and nondimensionalization as runhdg_ns_boussinesq.jl (buoyancy
velocity U = sqrt(gβΔT L); hot left wall θ = +1/2, cold right wall θ = -1/2,
insulated top/bottom, no-slip). The mesh is wall-clustered through the smooth
map m(ξ) = ξ - s·sin(2πξ)/(2π) (elements become curved isoparametric, which
the HDG solvers support), resolving the Ra^(-1/4) boundary layers.

The run animates the transient from the conductive state: boundary layers
form, plumes rise along the hot wall, and the stratified core fills in. The
hot-wall Nusselt number is logged every step for comparison with the
de Vahl Davis / Le Quéré benchmarks (Nu = 16.523 at Ra = 10⁷).

Usage:
    JULIA_LOAD_PATH='@;@cuda;@v#.#;@stdlib' julia +1.12 --project=. \
        examples/hdg/runhdg_ns_boussinesq_animation.jl [solve|render|all]

`solve` writes one Serialization snapshot per frame to ./output/boussinesq_anim/
(restart-friendly); `render` turns whatever snapshots exist into the GIF.
GPU is used when CUDA.jl is available in the environment stack (falls back to
the CPU backend of the same kernels otherwise).
=#
using TwoDG
using LinearAlgebra
using Printf
using Serialization

#%% Parameters
Ra = 1e7
Pr = 0.71
ν = sqrt(Pr / Ra)              # momentum diffusivity (buoyancy-velocity scaling)
κ = 1 / sqrt(Ra * Pr)          # thermal diffusivity

n = 48                         # n × n × 2 triangles
porder = 3
stretch = 0.5                  # wall-clustering strength of m(ξ)
Δt = 0.02
t_end = 20.0
nnewton = 1                    # 1 = semi-implicit Oseen linearization per step:
                               # at this Δt a second Newton pass changes Nu/KE
                               # by < 1e-3 (checked), and the final Nu is
                               # validated against the benchmark anyway
save_every = 3                 # steps between animation frames
τ = 1.0                        # HDG stabilization

outdir = joinpath(@__DIR__, "..", "..", "output", "boussinesq_anim")
gifpath = joinpath(@__DIR__, "..", "..", "figures", "hdg_ns_boussinesq_ra1e7.gif")

mode = isempty(ARGS) ? "all" : ARGS[1]

#%% Mesh: uniform square, then smooth wall clustering (curved elements)
function stretch_mesh!(mesh, s)
    m(ξ) = ξ - s * sin(2π * ξ) / (2π)
    for arr in (mesh.p, mesh.pcg)
        arr .= m.(arr)
    end
    mesh.dgnodes .= m.(mesh.dgnodes)
    return mesh
end

mesh = stretch_mesh!(mkmesh_square(n + 1, n + 1, porder, 0, 1), stretch)
master = Master(mesh, 3 * (porder + 1))
npl, nt = size(mesh.dgnodes, 1), size(mesh.t, 1)
nps, nf = porder + 1, size(mesh.f, 1)

# Boundary conditions (square tags: 1 bottom, 2 right, 3 top, 4 left)
dbc(p) = [0.0, 0.0]
tbc(p, tag) = tag == 4 ? (:d, 0.5) : tag == 2 ? (:d, -0.5) : (:n, 0.0)

#%% Hot-wall Nusselt number from q = ∇θ (as in runhdg_ns_boussinesq.jl)
function nusselt_hot(mesh, master, q)
    sh1d = master.sh1d[:, 1, :]
    sh1dx = master.sh1d[:, 2, :]
    Nu = 0.0
    for i in axes(mesh.f, 1)
        mesh.f[i, 4] == -4 || continue
        it = mesh.f[i, 3]
        lf = findfirst(x -> abs(x) == i, mesh.t2f[it, :])
        pp = master.perm[:, lf, 1]
        xξ = sh1dx' * mesh.dgnodes[pp, 1, it]
        yξ = sh1dx' * mesh.dgnodes[pp, 2, it]
        ds = sqrt.(xξ .^ 2 .+ yξ .^ 2)
        q1g = sh1d' * q[pp, 1, it]
        Nu -= sum(master.gw1d .* ds .* q1g)
    end
    return Nu
end

#%% Solve phase: operator splitting, everything batched, GPU when available
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
    logio = open(joinpath(outdir, "progress.log"), "a")   # explicit flush: the
    # stderr of a detached process is pipe-buffered and lost on a hard kill

    # resume from the last full-state checkpoint when present
    if isfile(statepath)
        st = deserialize(statepath)
        θ, u, Λ = st.θ, st.u, st.Λ
        t, step0, iframe = st.t, st.step, st.iframe
        hist_t, hist_nu, hist_ke = st.hist_t, st.hist_nu, st.hist_ke
        println(logio, "resuming from step $step0, t = $t"); flush(logio)
    else
        θ = [0.5 - mesh.dgnodes[k, 1, it] for k in 1:npl, it in 1:nt]
        u = zeros(npl, 2, nt)
        Λ = zeros(2 * nps * nf)
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

        src = zeros(npl, 2, nt)
        src[:, 2, :] .= θ
        uold = copy(u)
        for inner in 1:nnewton
            res = hdg_ns_step_batched(master, mesh, ν, dbc; τ, source=src,
                                      u, Λ, uold, dtinv, ArrayT, cache=nscache)
            u, Λ = res.u, res.Λ
            nscache = res.cache
        end

        t += Δt
        Nu = nusselt_hot(mesh, master, θres.q)
        ke = sum(u .^ 2)
        push!(hist_t, t); push!(hist_nu, Nu); push!(hist_ke, ke)
        all(isfinite, u) || error("solution diverged at t = $t")

        if step % save_every == 0
            iframe += 1
            serialize(joinpath(outdir, @sprintf("frame_%04d.bin", iframe)), (t=t, θ=θ))
        end
        if step % 50 == 0 || step == nsteps
            serialize(statepath, (; θ, u, Λ, t, step, iframe, hist_t, hist_nu, hist_ke))
        end
        if step % 25 == 0 || step == 1
            println(logio, @sprintf("step %4d/%d  t = %5.2f  Nu = %7.3f  KE = %.3e  (%.1f s/step)",
                                    step, nsteps, t, Nu, ke, (time() - wall0) / (step - step0)))
            flush(logio)
        end
    end
    serialize(joinpath(outdir, "history.bin"), (t=hist_t, nu=hist_nu, ke=hist_ke))
    println(logio, @sprintf("final hot-wall Nu = %.3f (Le Quéré benchmark at Ra = 1e7: 16.523)",
                            hist_nu[end]))
    close(logio)
end

#%% Render phase: Observable-driven per-element mesh plot -> GIF
# (the `using` must be a separate top-level expression from the code using
# Makie macros: a single if-block is lowered before the `using` executes)
if mode in ("render", "all")
    using CairoMakie
end

if mode in ("render", "all")
    frames = sort(filter(f -> startswith(f, "frame_"), readdir(outdir)))
    @info "rendering $(length(frames)) frames"
    θs = [deserialize(joinpath(outdir, f)) for f in frames]

    iobs = Observable(1)
    plot_field = @lift θs[$iobs].θ
    title_str = @lift @sprintf("Natural convection, Ra = 10⁷ (HDG on GPU)   t = %.2f", θs[$iobs].t)

    fig = Figure(size=(640, 560))
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y", title=title_str)
    faces = Matrix(mesh.tlocal[:, 1:3])
    for it in 1:nt
        mesh!(ax, mesh.dgnodes[:, :, it], faces,
              color=@lift($plot_field[:, it]), colormap=Reverse(:RdBu),
              colorrange=(-0.5, 0.5))
    end
    Colorbar(fig[1, 2], colormap=Reverse(:RdBu), limits=(-0.5, 0.5), label="θ")
    xlims!(ax, 0, 1); ylims!(ax, 0, 1)

    # record an mp4, then convert to GIF with a two-pass palette — the default
    # GIF encoder output is ~4-5x larger for smooth colormap fields
    mkpath(dirname(gifpath))
    mp4path = replace(gifpath, ".gif" => ".mp4")
    CairoMakie.record(fig, mp4path, eachindex(θs); framerate=24) do i
        i % 25 == 0 && @info "frame $i / $(length(θs))"
        iobs[] = i
    end
    @info "wrote $mp4path ($(round(filesize(mp4path) / 1e6, digits=1)) MB)"

    # 128 colors + no dither is the size lever: dithered smooth gradients are
    # high-entropy noise to GIF's LZW; a reduced palette with flat runs is ~15x
    # smaller at barely-visible banding (measured on this content)
    ffmpeg = Sys.which("ffmpeg") === nothing ?
        CairoMakie.Makie.FFMPEG_jll.ffmpeg() : `ffmpeg`
    vf = "fps=15,scale=600:-1:flags=lanczos,split[s0][s1];" *
         "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=none"
    run(`$ffmpeg -y -loglevel error -i $mp4path -vf $vf $gifpath`)
    @info "wrote $gifpath ($(round(filesize(gifpath) / 1e6, digits=1)) MB)"
end
