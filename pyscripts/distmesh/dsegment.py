import numpy as np

def dsegment(p, v):
    """
    d = dsegment(p, v)

    Parameters
    ----------
    p : array, shape (np, 2)
        points
    v : array, shape (nv, 2)
        vertices of a closed array, whose edges are v[0]..v[1],
        ... v[nv-2]..v[nv-1]

    Output
    ------
    ds : array, shape (np, nv-1)
        distance from each point to each edge
    """

    large = 1.0e+30
    dis = large * np.ones(p.shape[0])

    for i in range(v.shape[0]-1):
        ds = donesegment(p, v[i:i+2,:])
        dis = np.minimum(dis, donesegment(p, v[i:i+2,:]))

    return dis

def donesegment(p, v):
    d = np.diff(v, axis = 0)[0]
    sca0 = np.linalg.norm(d, axis = 0)
    sca = np.sum((p - np.tile(v[0,:],(p.shape[0],1))) * np.tile(d,(p.shape[0],1)), axis=1)

    ds = np.zeros(p.shape[0])

    ind0 = sca <= 0.0
    ds[ind0] = np.linalg.norm((p[ind0,:] - np.tile(v[0,:], (ind0.sum(),1))), axis = 1)

    ind1 = sca >= sca0
    ds[ind1] = np.linalg.norm((p[ind1,:] - np.tile(v[1,:], (ind1.sum(),1))), axis = 1)

    ind = np.logical_not(ind0) * np.logical_not(ind1)
    ds[ind] = np.linalg.norm((p[ind,:] - np.tile(v[0,:], (ind.sum(),1)) - np.tile(d, (ind.sum(),1)) * np.transpose(np.tile(sca[ind], (2,1))) / sca0**2), axis = 1)

    return ds

if __name__ == "__main__":
    p = np.array([[-1, 1], [0.5, 1], [0.75, 1], [3, 1]])
    v = np.array([[0.5, 0], [2, 0], [3, 0]])
    
    dis = dsegment(p, v)

    print(dis)