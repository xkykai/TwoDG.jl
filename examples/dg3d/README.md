# 3D DG examples

Explicit DG/LDG on tetrahedra (THREED_PLAN Phases C–D). Run from the repo
root, e.g.

```
julia --project=. examples/dg3d/run3d_convection_box.jl
```

Each script prints its error/rate table; with **WriteVTK.jl** loaded in the
environment it also writes high-order Lagrange `.vtu` files into
`examples/dg3d/output/` — open them in **ParaView** (curved elements render
natively; for the time series in `run3d_euler_pulse.jl`, open the
`pulse_..vtu` group and press play).

| Script | What it shows | Exact solution |
|---|---|---|
| `run3d_convection_box.jl` | entry-level: box mesh, DG transport, `p+1` rate | translated sine product |
| `run3d_euler_vortex.jl` | Euler + Roe through `DGProblem`/`solve`/`compute_dt`, `p+1` rate, Mach output | isentropic vortex (z-aligned axis) |
| `run3d_heat_sphere_octant.jl` | curved isoparametric tets: LDG heat in a ball octant (Bey `uniref` + sphere projection) | radial mode `j₀(πr) e^{-κπ²t}` |
| `run3d_euler_pulse.jl` | acoustic pulse in a slip-wall box, VTK time series via the solve callback | — (qualitative) |

GPU: pass `ArrayT = CuArray` to `solve` (with `using CUDA`) — same scripts,
same kernels; see `examples/dg/run_ka_cuda.jl` for the pattern.
