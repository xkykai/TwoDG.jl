"""
    interpolate(mesh, values)

Nodal interpolation of initial/reference data onto the DG nodes. `values` is
a collection with one entry per solution component; each entry is either a
number (constant field) or a function `(x, y) -> value`. Returns
`u (npl, nc, nt)` with `nc = length(values)`.
"""
function interpolate(mesh, values)
    nc = length(values)
    u = zeros(size(mesh.dgnodes, 1), nc, size(mesh.dgnodes, 3))
    for i in 1:nc
        if isa(values[i], Number)
            u[:, i, :] .= values[i]
        else
            u[:, i, :] .= values[i].(mesh.dgnodes[:, 1, :], mesh.dgnodes[:, 2, :])
        end
    end
    return u
end

"""
    initu(mesh, app, value)

Initialize the vector of unknowns `u (npl, app.nc, nt)` from per-component
constants or functions (see [`interpolate`](@ref), which this wraps).
"""
function initu(mesh, app, value)
    length(value) == app.nc ||
        throw(ArgumentError("expected $(app.nc) components, got $(length(value))"))
    return interpolate(mesh, value)
end
