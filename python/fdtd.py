import math
import numpy as np

def init_fdtd(tmax, nx, ny):
    _fict_ = np.zeros(tmax)
    ex = np.zeros((nx,ny))
    ey = np.zeros((nx,ny))
    hz = np.zeros((nx,ny))

    for i in range(tmax):
        _fict_[i] = np.float32(i)
    
    
    for i in range(nx):
        for j in range(ny):
            ex[i, j] = np.float32((i * (j + 1)) / nx)
            ey[i, j] = np.float32((i * (j + 2)) / ny)
            hz[i, j] = np.float32((i * (j + 3)) / nx)

    return ex, ey, hz, _fict_

def kernel_fdtd_2d(tmax, nx, ny, ex, ey, hz, _fict_):
    for t in range(tmax):
        for j in range(ny):
            ey[0, j] = _fict_[t]
        
        for i in range(1, nx):
            for j in range(ny):
                ey[i, j] = ey[i, j] - 0.5 * (hz[i, j] - hz[i-1, j])
        
        for i in range(nx):
            for j in range(1, ny):
                ex[i, j] = ex[i, j] - 0.5 * (hz[i, j] - hz[i, j-1])
        
        for i in range(nx - 1):
            for j in range(ny - 1):
                hz[i, j] = hz[i, j] - 0.7 * (ex[i, j+1] - ex[i, j] + ey[i+1, j] - ey[i, j])

