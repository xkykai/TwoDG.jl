using TwoDG
using CSV
using DataFrames
using CairoMakie
using LinearAlgebra

function mkmesh_duct(mesh, db, dt, H)
    p = mesh.p
    mesh.fcurved .= true
    mesh.tcurved .= true
    
    pnew = zeros(size(p))
    pnew[:,1] .= 3.0 .* p[:,1]
    
    ii = findall(x -> x <= 1 || x >= 2, pnew[:, 1])
    pnew[ii, 2] .= H .* p[ii, 2]
    
    ii = findall(x -> x > 1 && x < 2, pnew[:, 1])
    pnew[ii, 2] .= p[ii, 2] .* (H .- dt .* sin.(π .* (pnew[ii, 1] .- 1)).^2) .+ 
                    (1 .- p[ii, 2]) .* (db .* sin.(π .* (pnew[ii, 1] .- 1)).^2)
    
    mesh = TwoDG.Mesh(; p=pnew, mesh.t, mesh.f, mesh.t2f, mesh.fcurved, mesh.tcurved, mesh.porder, mesh.plocal, mesh.tlocal)
    
    fd_left(p) = abs(p[1])
    fd_right(p) = abs(p[1] .- 3)
    
    function fd_bottom(p, H, db)
        x, y = p
        if x <= 1 || x >= 2
            return abs(y)
        else
            return abs(y - ((1 - y/H) .* (db * sin(π * (x - 1))^2)))
        end
    end
    fd_bottom(p) = fd_bottom(p, H, db)
    
    function fd_top(p, H, dt)
        x, y = p
        if x <= 1 || x >= 2
            return abs(y - H)
        else
            return abs(y - (H - dt * sin(π * (x - 1))^2))
        end
    end
    fd_top(p) = fd_top(p, H, dt)
    
    fds = [fd_left, fd_right, fd_bottom, fd_top]
    
    mesh = createnodes(mesh, fds)

    return mesh
end