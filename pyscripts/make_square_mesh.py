#%%
import distmesh as dm
import numpy as np

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

m = int(sys.argv[1])
n = int(sys.argv[2])
parity = int(sys.argv[3])
directory_path = sys.argv[4]

def squaremesh(m=10, n=10, parity=0):
    """ 
    squaremesh 2-d regular triangle mesh generator for the unit square
    p, t = squaremesh(m, n, parity)
 
      p:         node positions (np,2)
      t:         triangle indices (nt,3)
      parity:    flag determining the the triangular pattern
                  flag = 0 (diagonals sw - ne) (default)
                  flag = 1 (diagonals nw - se)
    """
    # Create regular grid points
    x = np.linspace(0, 1, m)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    
    # Create points array
    p = np.column_stack((X.flatten(), Y.flatten()))
    
    # Create triangulation
    tri = []
    for i in range(n-1):
        for j in range(m-1):
            idx = i*m + j
            if parity == 0:
                # Southwest to Northeast diagonals
                tri.append([idx, idx+1, idx+m+1])
                tri.append([idx, idx+m+1, idx+m])
            else:
                # Northwest to Southeast diagonals
                tri.append([idx, idx+1, idx+m])
                tri.append([idx+1, idx+m+1, idx+m])
    
    # Convert triangulation to numpy array
    t = np.array(tri, dtype=int)

    return p, t

p, t = squaremesh(m, n, parity)

os.makedirs(directory_path, exist_ok=True)

np.savetxt(os.path.join(directory_path, f'p.csv'), p, delimiter=',')
np.savetxt(os.path.join(directory_path, f't.csv'), t, delimiter=',', fmt='%d')

#%%