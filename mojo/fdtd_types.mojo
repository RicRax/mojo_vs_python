from time import now
from matrix_types import Matrix
import benchmark

alias tmax = 10
alias nx = 10
alias ny = 10

fn kernel_fdtd_2d_types(tmax: Int, nx: Int, ny: Int, ex: Matrix, ey: Matrix, hz: Matrix, _fict_: Matrix):
    for t in range(tmax):
        for j in range(ny):
            ey[0,j] = _fict_[0,t]
        
        for i in range(1, nx):
            for j in range(ny):
                ey[i,j] = ey[i,j] - 0.5 * (hz[i,j] - hz[i-1,j])
        
        for i in range(nx):
            for j in range(1, ny):
                ex[i,j] = ex[i,j] - 0.5 * (hz[i,j] - hz[i,j-1])
        
        for i in range(nx - 1):
            for j in range(ny - 1):
                hz[i,j] = hz[i,j] - 0.7 * (ex[i,j+1] - ex[i,j] + ey[i+1,j] - ey[i,j])



fn benchmark_fdtd_types() -> object:
    var res = 0
    for i in range(10):
        var ex = Matrix[nx,ny]().rand()
        var ey = Matrix[nx,ny]().rand()
        var hz = Matrix[nx,ny]().rand()
        var _fict_ = Matrix[nx,ny]()


        var prev = now()
        kernel_fdtd_2d_types(tmax, nx, ny, ex, ey , hz , _fict_)
        var curr = now()
        res += curr - prev

        ex.data.free()
        ey.data.free()
        hz.data.free()
        _fict_.data.free()


    res = res // 10
    return res
