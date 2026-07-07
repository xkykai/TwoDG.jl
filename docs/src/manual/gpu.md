# GPU support

Every performance-critical path in TwoDG is written once, as
KernelAbstractions kernels over plain arrays, and runs unchanged on CPU
threads or a GPU. CUDA.jl is deliberately **not** a dependency: load it
yourself and pass a device array type.

```julia
using TwoDG, CUDA

sol = solve(prob, RK4(); dt = compute_dt(prob), tfinal = 1.0, ArrayT = CuArray)
```

`ArrayT = CuArray` works on [`DGProblem`](@ref) (the whole RK4 time loop),
[`HDGProblem`](@ref) with [`GMRES`](@ref) (batched local solves, assembly,
Krylov trace iterations, and recovery), and [`CGProblem`](@ref) with
[`ConjugateGradient`](@ref)/[`GMRES`](@ref) (the matrix-free iteration).
Any other KernelAbstractions backend array type works the same way.

## What moves to the device — and what doesn't

**Device:** DG residuals and RK4 stages ([`DGContext`](@ref) + pointwise
fluxes), LDG gradient/viscous kernels, HDG batched GEMMs and in-kernel LU
([`HDGBatch`](@ref)), the HDG trace matvec and block-Jacobi preconditioner
([`HDGSystem`](@ref)), the CG stiffness operator and Jacobi preconditioner.

**CPU (by design):** mesh generation and high-order node projection,
[`ReferenceElement`](@ref) construction, one-time geometry precomputation,
source-term evaluation (user closures), sparse direct factorizations
([`Direct`](@ref) — sparse LU/Cholesky is a poor GPU fit), and plotting.
Setup runs once; the device gets what iterates.

The same applies in 3D — see [3D in TwoDG](threed.md) for measured 3D
speedups and the quadrature/precision guidance that goes with them.

## Working with device data directly

The low-level structs are `Adapt`-able — everything is a plain array field:

```julia
using Adapt

ctx  = DGContext(master, mesh; T = Float32)
dctx = adapt(CuArray, ctx)                 # whole geometry cache on the GPU
phys_d = adapt(CuArray, phys)              # DGPhysics: equation + BCs + fluxes
u_d   = CuArray(Float32.(u0))
rk4_ka!(inviscid_residual!, dctx, phys_d, u_d, 0f0, dt, nstep)
```

## Precision

All solver structs are eltype-parametric; `T = Float32` runs end to end in
single precision, which matters on consumer GPUs (FP64 throughput is
typically 1/32–1/64 of FP32). Empirical guidance from the smoke benchmarks
(`examples/dg/run_ka_cuda.jl`):

- **Explicit DG**: Float32 is the right choice — residual noise matches the
  CPU's Float32 noise, and throughput wins grow with mesh size.
- **HDG GMRES**: use Float64 on the GPU. Single-precision Krylov iterations
  stagnate before tight tolerances on large trace systems; the matvec is
  memory-bound, so consumer-card FP64 throughput is not the bottleneck.
- **CG conjugate gradients**: Float64 matches the direct solve to ~1e-14;
  Float32 converges to ~1e-4 solution accuracy.

Small meshes are launch-overhead-bound — the GPU pays off at scale (tens of
thousands of elements), and for one-shot 2D solves the CPU sparse direct
path is often still the fastest option. Benchmark before assuming.

## Running the smoke benchmark

`examples/dg/run_ka_cuda.jl` exercises DG, LDG, HDG, and CG on the GPU with
correctness assertions and timings. CUDA.jl is best installed once into a
shared environment:

```
julia -e 'using Pkg; Pkg.activate("cuda"; shared=true); Pkg.add("CUDA")'
JULIA_LOAD_PATH='@;@cuda;@v#.#;@stdlib' julia --project=. examples/dg/run_ka_cuda.jl
```
