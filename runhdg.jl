using TwoDG
using CairoMakie

porder = 4
ngauss = 2 * (porder + 1)
boundary_refinement = 3

hdg_source(p) = ones(size(p, 1), 1)
dbc(p) = zeros(size(p, 1), 1)

kappa = 1
taud = 1
sizs = [0.25, 0.2, 0.1]
cs = [[1, 1], [10, 10], [100, 100]]

for siz in sizs
    mesh = mkmesh_circle(siz, porder, 1; boundary_refinement)
    master = Master(mesh, ngauss)
    
    mesh1 = make_circle_nodes(mesh.p, mesh.t, porder + 1, 1)
    master1 = Master(mesh1, ngauss)
    
    for c in cs
        @info "Running for size = $siz and c = $c"
        param = Dict(:kappa => kappa, :c => c, :taud => taud)

        u, q, uh = hdg_solve(master, mesh, hdg_source, dbc, param)
        ustarh = hdg_postprocess(master, mesh, master1, mesh1, u, q ./ kappa)

        fig = scaplot(mesh, u[:, 1, :], show_mesh=true, title="u")
        save("output/hdg_convdiff_u_size_$(siz)_k_$(kappa)_c_$(c[1])_$(c[2])_p_$(porder).png", fig, px_per_unit=4)

        fig = scaplot(mesh1, ustarh[:, 1, :], show_mesh=true, title="u*")
        save("output/hdg_convdiff_ustar_size_$(siz)_k_$(kappa)_c_$(c[1])_$(c[2])_p_$(porder).png", fig, px_per_unit=4)
    end
end