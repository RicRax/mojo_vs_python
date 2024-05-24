from time import now
from algorithm import vectorize
from matrix_types import Matrix
from algorithm import stencil
import benchmark

alias nelts = simdwidthof[DType.float32]() * 2
alias tmax = 10
alias nx = 10
alias ny = 10

fn kernel_fdtd_2d_vector(ex: Matrix, ey: Matrix, hz: Matrix, _fict_: Matrix):
    for t in range(tmax):
        for j in range(ny):
            ey[0,j] = _fict_[0,t]
        
        for i in range(1, nx):
            @parameter
            fn calc[nelts: Int](j : Int):
                ey.store(i, j, ey.load[nelts](i,j) - 0.5 * (hz.load[nelts](i,j) - hz.load[nelts](i - 1, j)))
            vectorize[calc, nelts, size = ny]()
        
        for i in range(nx):
            @parameter
            fn calc2[nelts: Int](j : Int):
                if (j != 0):
                    ex.store(i, j, ex.load[nelts](i,j) - 0.5 * (hz.load[nelts](i,j) - hz.load[nelts](i, j - 1)))
                    ex[i,j] = ex[i,j] - 0.5 * (hz[i,j] - hz[i,j-1])
            vectorize[calc2, nelts, size = ny]()
        
        for i in range(nx - 1):
            @parameter
            fn calc3[nelts: Int](j : Int):
                hz.store(i, j, hz.load[nelts](i,j) - 0.7 * (ex.load[nelts](i, j + 1) - ex.load[nelts](i,j) + ey.load[nelts](i+1,j)))
            vectorize[calc3, nelts, size = ny - 1]()



fn benchmark_fdtd_vector() -> object:
    var res = 0
    for i in range(10):
        var ex = Matrix[nx,ny]().rand()
        var ey = Matrix[nx,ny]().rand()
        var hz = Matrix[nx,ny]().rand()
        var _fict_ = Matrix[nx,ny]()


        var prev = now()
        kernel_fdtd_2d_vector(ex, ey , hz , _fict_)
        var curr = now()
        res += curr - prev

        ex.data.free()
        ey.data.free()
        hz.data.free()
        _fict_.data.free()


    res = res // 10
    return res
