using SparseArrays
using LinearAlgebra

"""
    cg_solve(mesh, master, source, param) -> (uh, energy)

Solve the steady convection-diffusion(-reaction) equation with continuous
Galerkin finite elements and homogeneous Dirichlet boundaries, on triangles
or tetrahedra. `mesh` must carry the CG numbering (see `cgmesh`); the source
term is called with the coordinates splatted (`source(x, y)` in 2D,
`source(x, y, z)` in 3D); `param` is a named tuple `(; κ, c, s)` with
diffusivity `κ`, convective velocity `c` (length `Dim`), and reaction
coefficient `s`.

Element matrices are built batched ([`cg_element_system`](@ref)), assembled
from triplets ([`cg_assemble`](@ref)), and factorized directly: Cholesky
when the operator is SPD (no convection, `s ≥ 0`), sparse LU otherwise.
For an iterative, GPU-capable solve see [`cg_parsolve`](@ref).

Returns the solution `uh (npl, nt)` in DG (element-local) numbering, ready
for `scaplot`, and the discrete energy `½ uᵀKu - uᵀF`.

The high-level equivalent is `solve(CGProblem(equation, mesh; source))`.
"""
function cg_solve(mesh, master, source, param)
    ae, fe, dirichlet, symmetric = cg_element_system(mesh, master, source, param)
    K, F = cg_assemble(ae, fe, mesh.tcg, dirichlet)

    u = if symmetric && param.s >= 0
        cholesky(Symmetric(K)) \ F
    else
        K \ F
    end
    energy = 0.5 * dot(u, K, u) - dot(u, F)

    # Output uh (DG format) to make it compatible with scaplot
    uh = Matrix{eltype(u)}(undef, size(mesh.dgnodes, 1), size(mesh.dgnodes, 3))
    for e in axes(mesh.tcg, 1)
        uh[:, e] .= u[mesh.tcg[e, :]]
    end

    return uh, energy
end
