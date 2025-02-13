#%%
import distmesh as dm
import numpy as np

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

size = float(sys.argv[1])
directory_path = sys.argv[2]

def make_circle_mesh(siz=0.4):
    fd = lambda p: np.sqrt((p**2).sum(1))-1.0
    p, t = dm.distmesh2d(fd, dm.huniform, siz, (-1,-1,1,1))
    return p, t

# size = 0.6
p, t = make_circle_mesh(size)

os.makedirs(directory_path, exist_ok=True)

np.savetxt(os.path.join(directory_path, f'p.csv'), p, delimiter=',')
np.savetxt(os.path.join(directory_path, f't.csv'), t, delimiter=',', fmt='%d')

#%%