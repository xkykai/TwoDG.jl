using TwoDG
using BenchmarkTools
using LinearAlgebra
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 30
BLAS.set_num_threads(1)

porder = 10
m = n = 21
parity = 0
nodetype = 1

mesh = mkmesh_square(m, n, porder, parity, nodetype)
master = Master(mesh, 2 * porder)

kappa = 1
c = [10, 10]
taud = 1
param = Dict(:kappa => kappa, :c => c, :taud => taud)
hdg_source(p) = 10 * ones(size(p, 1), 1)
dbc(p) = zeros(size(p, 1), 1)

restart = 100
@benchmark uh, qh, uhath = hdg_parsolve(master, mesh, hdg_source, dbc, param; restart)