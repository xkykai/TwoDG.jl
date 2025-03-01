import distmesh as dm
import numpy as np
import sys
import os

# Get current script directory and add parent directory to path for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Get mesh size and output directory from command line arguments
size = float(sys.argv[1])
directory_path = sys.argv[2]

def make_circle_mesh(siz=0.4):
    # Define a signed distance function for a circle with radius 1
    # Points inside the circle have negative values, points on the circle have value 0,
    # and points outside have positive values
    fd = lambda p: np.sqrt((p**2).sum(1))-1.0
    
    # Generate the mesh using distmesh2d
    # fd: the distance function defining the circle
    # dm.huniform: uniform mesh size function (constant element size)
    # siz: controls the fineness of the mesh (smaller value = finer mesh)
    # (-1,-1,1,1): bounding box coordinates (xmin, ymin, xmax, ymax)
    p, t = dm.distmesh2d(fd, dm.huniform, siz, (-1,-1,1,1))
    
    # Return points and triangles
    # p: node/vertex coordinates (n×2 array where n is number of points)
    # t: triangle indices (m×3 array where m is number of triangles)
    return p, t

# Generate the mesh with specified size
p, t = make_circle_mesh(size)

# Create output directory if it doesn't exist
os.makedirs(directory_path, exist_ok=True)

# Save points coordinates to p.csv with comma delimiter
np.savetxt(os.path.join(directory_path, f'p.csv'), p, delimiter=',')

# Save triangle indices to t.csv with comma delimiter
# fmt='%d' ensures triangle indices are saved as integers
np.savetxt(os.path.join(directory_path, f't.csv'), t, delimiter=',', fmt='%d')