"""
    interpolate(mesh, values)

Nodal interpolation of initial/reference data onto the DG nodes. `values` is
a collection with one entry per solution component; each entry is either a
number (constant field) or a function of the coordinates (`(x, y) -> value`
in 2D, `(x, y, z) -> value` in 3D). Returns `u (npl, nc, nt)` with
`nc = length(values)`.
"""
function interpolate(mesh, values)
    nc = length(values)
    Dim = size(mesh.dgnodes, 2)
    coords = ntuple(d -> view(mesh.dgnodes, :, d, :), Dim)
    u = zeros(size(mesh.dgnodes, 1), nc, size(mesh.dgnodes, 3))
    for i in 1:nc
        if isa(values[i], Number)
            u[:, i, :] .= values[i]
        else
            u[:, i, :] .= values[i].(coords...)
        end
    end
    return u
end

"""
    initu(mesh, nc::Integer, value)

Initialize the vector of unknowns `u (npl, nc, nt)` from per-component
constants or functions (see [`interpolate`](@ref), which this wraps). Pass
`nvariables(equation)` for `nc`.
"""
function initu(mesh, nc::Integer, value)
    length(value) == nc ||
        throw(ArgumentError("expected $nc components, got $(length(value))"))
    return interpolate(mesh, value)
end
