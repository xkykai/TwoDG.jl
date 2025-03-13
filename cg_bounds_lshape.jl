using TwoDG
using CairoMakie
using LinearAlgebra
using CSV
using DataFrames

m = 5
porder = 3

mesh = mkmesh_lshape(m, porder)
master = Master(mesh, 3*porder)

meshplot_curved(mesh, nodes=true)
κ = 1
c = [0, 0]
s = 0
param = (; κ, c, s)

source(x, y) = 1

uh, energy_upper, u = cg_solve(mesh, master, source, param)
scaplot(mesh, uh, show_mesh=true)

guh = grad_u(master, mesh, uh)
scaplot(mesh, guh[:, 1, :], show_mesh=true)
qn, qn0 = equilibrate(master, mesh, guh, source)

function compute_qn_lineintegral(master, mesh, qn)
    qn_lineintegrals = zeros(size(mesh.t, 1))
    for it in 1:size(mesh.t, 1)
        t2f = mesh.t2f[it, :]
        for local_face_idx in eachindex(t2f)
            face_idx = t2f[local_face_idx]
            node_numbers = master.perm[:, local_face_idx, 1]
            element_coordinates = mesh.dgnodes[node_numbers, :, it]

            if face_idx < 0
                q = -qn[:, -face_idx][end:-1:1]
            else
                q = qn[:, face_idx]
            end
            
            for k in eachindex(master.gw1d)
                τ = master.sh1d[:, 2, k]' * element_coordinates  # Tangent vector at quadrature point
                τ_norm = sqrt(sum(τ.^2))
                qj = sum(master.sh1d[:, 1, k] .* q)

                qn_lineintegrals[it] += qj * master.gw1d[k] * τ_norm
            end

        end
    end
    return qn_lineintegrals
end

qn_lineintegrals = compute_qn_lineintegral(master, mesh, qn)

function compute_source_integral(master, mesh, f=1)
    f_integrals = zeros(size(mesh.t, 1))

    for it in 1:size(mesh.t, 1)
        element_coordinates = mesh.dgnodes[:, :, it]
        for k in eachindex(master.gwgh)
            J = master.shap[:, 2:3, k]' * element_coordinates
            detJ = det(J)

            f_integrals[it] += detJ * f * master.gwgh[k]
        end
    end
    return f_integrals
end

f_integrals = compute_source_integral(master, mesh)

q, energy_lowerq = reconstruct(master, mesh, source, qn)

#%%
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Element", title="Equilibrated flux anomaly (∫ f dΩ + ∫ qₙ dS)", ylabel="Anomaly")
scatter!(ax, 1:size(mesh.t, 1), f_integrals .+ qn_lineintegrals)
ylims!(ax, (-3e-16, 3e-16))
display(fig)
# save("./output/equilibrated_flux_anomaly_lshape.pdf", fig)
#%%
function compute_lower_bound(mesh, master, q)
    energy_lower = 0.
    for it in 1:size(mesh.t, 1)
        element_coordinates = mesh.dgnodes[:, :, it]
        for k in eachindex(master.gwgh)
            J = master.shap[:, 2:3, k]' * element_coordinates
            detJ = det(J)
            qk = [sum(master.shap[:, 1, k] .* q[:, 1, it]), sum(master.shap[:, 1, k] .* q[:, 2, it])]
            energy_lower += detJ * (qk ⋅ qk) * master.gwgh[k]
        end
    end
    energy_lower *= -0.5
    return energy_lower
end

energy_lower = compute_lower_bound(mesh, master, q)
@info "energy upper $(energy_upper), energy lower $(energy_lower), energy lower q $(energy_lowerq)"
#%%

