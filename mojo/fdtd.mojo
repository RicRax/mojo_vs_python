from matrix import matrix_init
from time import now
from matrix_types import Matrix
import benchmark

def kernel_fdtd_2d(tmax: Int, nx: Int, ny: Int, ex, ey, hz, _fict_):
    for t in range(tmax):
        for j in range(ny):
            ey[0][j] = _fict_[0][t]
        
        for i in range(1, nx):
            for j in range(ny):
                ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i-1][j])
        
        for i in range(nx):
            for j in range(1, ny):
                ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j-1])
        
        for i in range(nx - 1):
            for j in range(ny - 1):
                hz[i][j] = hz[i][j] - 0.7 * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j])



def benchmark_fdtd(tmax: Int, nx: Int, ny: Int):
    var res = 0
    for i in range(10):
        ex = matrix_init(nx,ny)
        ey = matrix_init(nx,ny)
        hz = matrix_init(nx,ny)
        _fict_ = matrix_init(nx,ny)

        for i in range(nx):
            ex_row = object([])
            ey_row = object([])
            hz_row = object([])
            for j in range(ny):
                ex_row.append((i * (j + 1)) / nx)
                ey_row.append((i * (j + 2)) / ny)
                hz_row.append((i * (j + 3)) / nx)
            ex.append(ex_row)
            ey.append(ey_row)
            hz.append(hz_row)


        _fict_row = object([])
        for i in range(tmax):
            _fict_row.append(0)
        _fict_.append(_fict_row)

        var prev = now()
        kernel_fdtd_2d(tmax, nx, ny, ex, ey , hz , _fict_)
        var curr = now()
        res += curr - prev

        _ = (ex, ey, hz, _fict_)
    
    res = res // 10
    return res

    