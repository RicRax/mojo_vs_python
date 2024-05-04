from matrix import matrix_init
from matrix_types import Matrix
import benchmark

def kernel_fdtd_2d(tmax, nx: Int, ny: Int, ex, ey, hz, _fict_):
    for t in range(tmax):
        for j in range(ny):
            ey[0][j] = _fict_[t]
        
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
    ex = matrix_init(nx,ny)
    ey = matrix_init(nx,ny)
    hz = matrix_init(nx,ny)
    _fict_ = matrix_init(nx,ny)

    for i in range(ni):
        a_row = object([])
        for j in range(nk):
            a_row.append((i * j) / ni)
        A.append(a_row)

    tmp_row = object([])
    for i in range(ny):
        tmp_row.append(0.0)
    tmp.append(tmp_row)

    @parameter
    fn test_fn():
        try:
            _ = kernel_3mm(ni, nj, nk, nl, nm, A, B, C, D, E, F, G )
        except:
            print("error occurred")

    var secs = benchmark.run[test_fn](max_runtime_secs=0.5).mean()
    _ = (A, B, C, D, E, F, G)
    return secs
    