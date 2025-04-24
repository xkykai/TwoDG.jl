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
boundary_refinement = float(sys.argv[2])
directory_path = sys.argv[3]

def make_circle_mesh(siz=0.4, boundary_refinement=2.0):
    # Define a signed distance function for a circle with radius 1
    fd = lambda p: np.sqrt((p**2).sum(1))-1.0
    
    # Create a custom mesh size function
    # This makes elements smaller near the boundary (where fd(p) is close to 0)
    def size_function(p):
        # Get distance from boundary (absolute value of signed distance)
        dist_from_boundary = np.abs(fd(p))
        # Calculate element size that's smaller near boundary
        # and gradually increases away from boundary
        h = siz * (0.2 + 0.8 * np.tanh(boundary_refinement * dist_from_boundary))
        return h
    
    # Generate the mesh using the custom size function instead of huniform
    p, t = dm.distmesh2d(fd, size_function, siz*0.2, (-1,-1,1,1))
    
    return p, t

# Generate the mesh with specified size
p, t = make_circle_mesh(size, boundary_refinement)

# Create output directory if it doesn't exist
os.makedirs(directory_path, exist_ok=True)

# Save points coordinates to p.csv with comma delimiter
np.savetxt(os.path.join(directory_path, f'p.csv'), p, delimiter=',')

# Save triangle indices to t.csv with comma delimiter
# fmt='%d' ensures triangle indices are saved as integers
np.savetxt(os.path.join(directory_path, f't.csv'), t, delimiter=',', fmt='%d')