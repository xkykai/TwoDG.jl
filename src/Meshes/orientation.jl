# The face-orientation vocabulary shared by the mesh connectivity (mkt2f/t2o),
# the reference element (perm), and the HDG trace numbering (elcon): which
# vertices span each local face, the symmetry group of the face simplex, and
# the detection of an element's orientation code relative to a face's stored
# traversal. Defined once, here — Masters imports it.

"""
    norient(Dim) -> Int

Number of relative orientations in which two elements can see a shared face:
the order of the symmetry group of the `Dim-1` face simplex — 1 in 1D (point
face), 2 in 2D (segment: identity + reversal), 6 in 3D (triangle: 3 rotations
× optional reflection).
"""
norient(Dim::Integer) = (1, 2, 6)[Dim]

"""
    face_vertices(::Val{Dim}, j) -> NTuple{Dim, Int}

Local vertices of local face `j` (the face opposite vertex `j`, where
barycentric coordinate `j` vanishes), in the traversal that makes the face
outward-oriented for the reference element: counterclockwise edges in 2D,
right-hand-rule outward triangles in 3D.
"""
face_vertices(::Val{2}, j::Integer) = ((2, 3), (3, 1), (1, 2))[j]
face_vertices(::Val{1}, j::Integer) = (j == 1 ? (2,) : (1,))
face_vertices(::Val{3}, j::Integer) = ((2, 3, 4), (1, 4, 3), (1, 2, 4), (1, 3, 2))[j]

"""
    orientation_permutations(::Val{Dim}) -> NTuple{norient, NTuple{Dim, Int}}

The symmetry group of the `Dim`-element's face simplex, as permutations of
the face's vertex positions; orientation code `o` refers to the `o`-th entry.
2D face = segment: identity, reversal. 3D face = triangle: the 3 rotations,
then the 3 reflected traversals.
"""
orientation_permutations(::Val{2}) = ((1, 2), (2, 1))
orientation_permutations(::Val{1}) = ((1,),)
orientation_permutations(::Val{3}) =
    ((1, 2, 3), (2, 3, 1), (3, 1, 2), (1, 3, 2), (3, 2, 1), (2, 1, 3))

"""
    face_orientation(stored, mine, ::Val{Dim}) -> Int

Orientation code `o` of an element's face traversal `mine` (tuple of global
vertex ids, the element's outward ordering, `Dim` of them) relative to the
face's canonical stored traversal `stored`: the index of the face-simplex
symmetry `σ` with `σ[j] = position of mine[j] in stored`, so that
`master.perm[:, s, o]` lists the element's volume nodes in the canonical
face-node order.
"""
@inline function face_orientation(stored::NTuple{N, Int}, mine::NTuple{N, Int},
                                  ::Val{Dim}) where {N, Dim}
    σ = ntuple(j -> findfirst(==(mine[j]), stored)::Int, Val(N))
    o = findfirst(==(σ), orientation_permutations(Val(Dim)))
    o === nothing &&
        error("faces $stored and $mine do not share the same vertex set — nonconforming mesh?")
    return o
end
