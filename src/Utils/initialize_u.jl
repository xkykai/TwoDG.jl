"""
initu initialize vector of unknowns
 mesh:             mesh structure
 app:              application structure
 value[app.nc]:    list containing
                   when value[i] is a float u[:,app.nc,:] = value[i]
                   when value[i] is a function,
                                u[:,app.nc,:] = value[i](mesh.dgnodes)
u[npl,app.nc,nt]: scalar function to be plotted
"""
function initu(mesh, app, value)
    u = zeros(size(mesh.dgnodes, 1), app.nc, size(mesh.dgnodes, 3))
    for i in 1:app.nc
        if isa(value[i], Number)
            u[:,i,:] .= value[i]
        else
            u[:,i,:] = value[i](mesh.dgnodes)
        end
    end
    return u
end