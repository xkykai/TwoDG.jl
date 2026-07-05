# Package extension: solution output as high-order Lagrange VTK cells for
# ParaView (THREED_PLAN D10). Loaded automatically when both TwoDG and
# WriteVTK are in the environment (`using TwoDG, WriteVTK`).
module TwoDGWriteVTKExt

using TwoDG
using TwoDG.Meshes: Mesh
using TwoDG.Masters: shape2d, shape3d, localpnts, localpnts3d
using TwoDG.Plotting: Plotting
using TwoDG.Equations: varnames
using WriteVTK

# ---------------------------------------------------------------------------
# VTK Lagrange point orderings (vtkLagrangeTriangle / vtkLagrangeTetra):
# vertices, then edge points (edge by edge, first→second vertex), then face
# points (face by face, recursively numbered), then interior points
# (recursively numbered). Generated constructively as barycentric index
# tuples — index position d is the lattice weight of the cell's vertex d —
# and matched against the mesh's own node lattice, so there is no
# hand-tabulated index magic (same policy as the face `perm` tables).

function vtk_triangle_lattice(n)
    n == 0 && return [(0, 0, 0)]
    pts = NTuple{3, Int}[(n, 0, 0), (0, n, 0), (0, 0, n)]
    for a in 1:(n - 1); push!(pts, (n - a, a, 0)); end     # edge V1→V2
    for a in 1:(n - 1); push!(pts, (0, n - a, a)); end     # edge V2→V3
    for a in 1:(n - 1); push!(pts, (a, 0, n - a)); end     # edge V3→V1
    for q in (n >= 3 ? vtk_triangle_lattice(n - 3) : NTuple{3, Int}[])
        push!(pts, (q[1] + 1, q[2] + 1, q[3] + 1))         # interior, recursive
    end
    return pts
end

# tet edges and faces in VTK order (1-based; VTK: edges 01,12,20,03,13,23 and
# faces 013,123,203,021)
const TET_EDGES = ((1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4))
const TET_FACES = ((1, 2, 4), (2, 3, 4), (3, 1, 4), (1, 3, 2))

function vtk_tet_lattice(n)
    n == 0 && return [(0, 0, 0, 0)]
    at(pairs...) = ntuple(d -> get(Dict(pairs...), d, 0), 4)
    pts = NTuple{4, Int}[at(1 => n), at(2 => n), at(3 => n), at(4 => n)]
    for (a, b) in TET_EDGES, s in 1:(n - 1)
        push!(pts, at(a => n - s, b => s))
    end
    for (a, b, c) in TET_FACES, q in (n >= 3 ? vtk_triangle_lattice(n - 3) : NTuple{3, Int}[])
        push!(pts, at(a => q[1] + 1, b => q[2] + 1, c => q[3] + 1))
    end
    for q in (n >= 4 ? vtk_tet_lattice(n - 4) : NTuple{4, Int}[])
        push!(pts, ntuple(d -> q[d] + 1, 4))
    end
    return pts
end

# permutation: VTK point slot -> row of the uniform barycentric lattice
function vtk_permutation(unif, porder, ::Val{Dim}) where {Dim}
    lattice = Dim == 2 ? vtk_triangle_lattice(porder) : vtk_tet_lattice(porder)
    lookup = Dict(ntuple(d -> round(Int, unif[i, d] * porder), Dim + 1) => i
                  for i in axes(unif, 1))
    perm = [lookup[q] for q in lattice]
    length(unique(perm)) == size(unif, 1) ||
        error("VTK Lagrange ordering did not produce a permutation (internal error)")
    return perm
end

# ---------------------------------------------------------------------------

function Plotting.save_vtk(mesh::Mesh, u::AbstractArray{<:Any, 3}, filename::AbstractString;
                           names=nothing)
    Dim = size(mesh.dgnodes, 2)
    porder = mesh.porder
    npl, _, nt = size(mesh.dgnodes)
    nc = size(u, 2)
    size(u, 1) == npl && size(u, 3) == nt ||
        throw(ArgumentError("field size $(size(u)) does not match the mesh ($(npl)×nc×$(nt))"))
    fieldnames = names === nothing ? ["u$c" for c in 1:nc] : collect(String.(names))
    length(fieldnames) == nc ||
        throw(ArgumentError("expected $nc component names, got $(length(fieldnames))"))

    # VTK Lagrange cells live on the uniform parametric lattice; resample the
    # geometry and the solution there (a no-op matrix when the mesh's nodes
    # are already uniform)
    unif = Dim == 2 ? localpnts(porder, 0)[1] : localpnts3d(porder, 0)
    V = Dim == 2 ? shape2d(porder, mesh.plocal, unif[:, 2:3])[:, 1, :] :
                   shape3d(porder, mesh.plocal, unif[:, 2:4])[:, 1, :]

    perm = vtk_permutation(unif, porder, Val(Dim))
    celltype = Dim == 2 ? VTKCellTypes.VTK_LAGRANGE_TRIANGLE :
                          VTKCellTypes.VTK_LAGRANGE_TETRAHEDRON

    points = zeros(3, npl * nt)
    data = [zeros(npl * nt) for _ in 1:nc]
    cells = Vector{MeshCell{VTKCellType, Vector{Int}}}(undef, nt)
    for e in 1:nt
        rows = (e - 1) * npl .+ (1:npl)
        points[1:Dim, rows] .= (V' * mesh.dgnodes[:, :, e])'
        for c in 1:nc
            data[c][rows] .= V' * u[:, c, e]
        end
        cells[e] = MeshCell(celltype, (e - 1) * npl .+ perm)
    end

    return vtk_grid(filename, points, cells) do vtk
        for c in 1:nc
            vtk[fieldnames[c]] = data[c]
        end
    end
end

Plotting.save_vtk(mesh::Mesh, u::AbstractMatrix, filename::AbstractString; kwargs...) =
    Plotting.save_vtk(mesh, reshape(u, size(u, 1), 1, size(u, 2)), filename; kwargs...)

function Plotting.save_vtk(sol::Union{TwoDG.Interface.DGSolution,
                                      TwoDG.Interface.HDGSolution,
                                      TwoDG.Interface.CGSolution},
                           filename::AbstractString)
    eq = sol.prob.equation
    names = size(sol.u, 2) == length(varnames(eq)) ? varnames(eq) : nothing
    return Plotting.save_vtk(sol.prob.mesh, sol.u, filename; names)
end

end # module
