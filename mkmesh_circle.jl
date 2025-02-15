using TwoDG
using CSV
using DataFrames
using CairoMakie

p, t = make_circle_mesh(0.6)

p_unique, t_unique = fixmesh(p, t)

meshscatter(p_unique[:, 1], p_unique[:, 2])

t_unique
# unique_faces, row_counts = mkt2f(t_unique)

t_test = [
    4 6 3;
    9 6 8;
    8 6 4;
    1 3 5;
    5 3 6;
    2 3 1;
    4 3 2;
    7 6 9;
    7 5 6;
]
f, t2f = mkt2f(t_unique)

f_test, t2f_test = mkt2f(t_test)

boundary(p) = sqrt.(sum(p.^2, dims=2)) .> 1 - 2e-2

bndexpr = [boundary]

f_boundary = setbndnbrs(p_unique, f, bndexpr)

