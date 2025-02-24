using ForwardDiff
using LinearAlgebra
using TwoDG.Utils: newton_raphson

struct Mesh{P, T, F, TF, FC, TC, PO, PL, TL, DG}
    p::P
    t::T
    f::F
    t2f::TF
    fcurved::FC
    tcurved::TC
    porder::PO
    plocal::PL
    tlocal::TL
    dgnodes::DG
    
    Mesh(p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal) = new{typeof(p), typeof(t), typeof(f), typeof(t2f), 
                                                                       typeof(fcurved), typeof(tcurved), typeof(porder),
                                                                       typeof(plocal), typeof(tlocal), Nothing}(
                                                                       p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal, nothing)

    Mesh(p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal, dgnodes) = new{typeof(p), typeof(t), typeof(f), typeof(t2f), 
                                                                                typeof(fcurved), typeof(tcurved), typeof(porder),
                                                                                typeof(plocal), typeof(tlocal), typeof(dgnodes)}(
                                                                                p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal, dgnodes)

    Mesh(p, t, porder, plocal, tlocal) = new{typeof(p), typeof(t), Nothing, Nothing, Nothing, Nothing, typeof(porder),
                                             typeof(plocal), typeof(tlocal), Nothing}(p, t, nothing, nothing, nothing, nothing, porder, plocal, tlocal, nothing)

    
    Mesh(mesh::Mesh, dgnodes) = new{typeof(mesh.p), typeof(mesh.t), typeof(mesh.f), typeof(mesh.t2f), 
                                    typeof(mesh.fcurved), typeof(mesh.tcurved), typeof(mesh.porder),
                                    typeof(mesh.plocal), typeof(mesh.tlocal), typeof(dgnodes)}(
                                    mesh.p, mesh.t, mesh.f, mesh.t2f, mesh.fcurved, mesh.tcurved, mesh.porder, mesh.plocal, mesh.tlocal, dgnodes)
end

function barycentric_to_cartesian(λ, v₁, v₂, v₃)
    T = hcat(v₂ .- v₁, v₃ .- v₁)
    return T * λ .+ v₁
end

autodiff(f) = x -> ForwardDiff.derivative(f, x)

function project_to_boundary(distance_function, x₀, s=0.1)
    grad = ForwardDiff.gradient(distance_function, x₀)
    grad_norm = grad / norm(grad)
    fd_linedirection(s) = distance_function(x₀ .+ s .* grad_norm)
    s = newton_raphson(fd_linedirection, autodiff(fd_linedirection), s)
    return x₀ .+ s .* grad_norm
end

function isvertex(λ)
    λ₁ = λ[2]
    λ₂ = λ[3]
    return (λ₁ == 0 || λ₁ == 1) && (λ₂ == 0 || λ₂ == 1)
end

function isedge(λ)
    return any(λ .== 0)
end

function edge_number(λ)
    if λ[3] == 0
        return 1
    elseif λ[2] == 0
        return 3
    else
        return 2
    end
end

function iscurvededge(λ, mesh, vn₁, vn₂, vn₃, it)
    all_curved_faces = mesh.f[mesh.fcurved, :]
    row_number = findfirst(x -> x == it, all_curved_faces[:, 3])
    curved_face = all_curved_faces[row_number, :]

    Eₙ = edge_number(λ)
    if Eₙ == 1
        return (vn₁ == curved_face[1] && vn₂ == curved_face[2]) || (vn₁ == curved_face[2] && vn₂ == curved_face[1])
    elseif Eₙ == 2
        return (vn₂ == curved_face[1] && vn₃ == curved_face[2]) || (vn₂ == curved_face[2] && vn₃ == curved_face[1])
    else
        return (vn₃ == curved_face[1] && vn₁ == curved_face[2]) || (vn₃ == curved_face[2] && vn₁ == curved_face[1])
    end

end

function iscurvedboundary(λ, mesh, vn₁, vn₂, vn₃, it)
    Eₙ = edge_number(λ)
    i_bnd = findfirst(x -> x < 0, mesh.f[:, 4])
    all_curved_boundaries = mesh.f[i_bnd:end, :]
    row_number = findfirst(x -> x == it, all_curved_boundaries[:, 3])

    if row_number === nothing
        return false
    elseif Eₙ == 1
        return (vn₁ == all_curved_boundaries[row_number, 1] && vn₂ == all_curved_boundaries[row_number, 2]) || (vn₁ == all_curved_boundaries[row_number, 2] && vn₂ == all_curved_boundaries[row_number, 1])
    elseif Eₙ == 2
        return (vn₂ == all_curved_boundaries[row_number, 1] && vn₃ == all_curved_boundaries[row_number, 2]) || (vn₂ == all_curved_boundaries[row_number, 2] && vn₃ == all_curved_boundaries[row_number, 1])
    else
        return (vn₃ == all_curved_boundaries[row_number, 1] && vn₁ == all_curved_boundaries[row_number, 2]) || (vn₃ == all_curved_boundaries[row_number, 2] && vn₁ == all_curved_boundaries[row_number, 1])
    end
end

function get_boundary_number(mesh, it)
    i_bnd = findfirst(x -> x < 0, mesh.f[:, 4])
    all_curved_boundaries = mesh.f[i_bnd:end, :]
    row_number = findfirst(x -> x == it, all_curved_boundaries[:, 3])
    return -all_curved_boundaries[row_number, 4]
end

function project_vertex_to_boundary!(mesh::Mesh, distance_functions::Union{Nothing, Vector})
    if distance_functions !== nothing
        i_bnd = findfirst(x -> x < 0, mesh.f[:, 4])
        n_curves = length(Set(mesh.f[i_bnd:end, 4]))
        all_curved_faces = mesh.f[i_bnd:end, :]

        for i in 1:n_curves
            fd = distance_functions[i]
            abs_fd(p) = abs(fd(p))
            curved_faces = all_curved_faces[all_curved_faces[:, 4] .== -i, :]
            unique_curved_nodes = Dict{Int, Nothing}()
            for node in curved_faces[:, 1:2]
                unique_curved_nodes[node] = nothing
            end

            for node in keys(unique_curved_nodes)
                node_coords = mesh.p[node, :]
                mesh.p[node, :] .= project_to_boundary(abs_fd, node_coords, 0.1)
            end
        end
    end
end

"""
createdgnodes computes the coordinates of the dg nodes.
dgnodes=createnodes(mesh,fd)

   mesh:      mesh data structure
   fd:        distance function d(x,y)
   dgnodes:   triangle indices (nplx2xnt). the nodes on 
              the curved boundaries are projected to the
              true boundary using the distance function fd
"""
function createnodes(mesh, fd=nothing)
    npl = size(mesh.plocal, 1)
    nt = size(mesh.t, 1)

    project_vertex_to_boundary!(mesh, fd)

    dgnodes = zeros(npl, 2, nt)

    for it in axes(dgnodes, 3)
        vn₁ = mesh.t[it, 1]
        vn₂ = mesh.t[it, 2]
        vn₃ = mesh.t[it, 3]

        v₁ = mesh.p[vn₁, :]
        v₂ = mesh.p[vn₂, :]
        v₃ = mesh.p[vn₃, :]

        iscurved_triangle = mesh.tcurved !== nothing && mesh.tcurved[it]
        for ipl in axes(dgnodes, 1)
            λ = mesh.plocal[ipl, :]
            x = barycentric_to_cartesian(λ[2:3], v₁, v₂, v₃)
            dgnodes[ipl, :, it] .= x
            if fd !== nothing && iscurved_triangle && !isvertex(λ) && isedge(λ) && iscurvedboundary(λ, mesh, vn₁, vn₂, vn₃, it)
                fdn = get_boundary_number(mesh, it)
                x = project_to_boundary(fd[fdn], x)
                dgnodes[ipl, :, it] .= x
            end
        end
    end

    return Mesh(mesh, dgnodes)
end