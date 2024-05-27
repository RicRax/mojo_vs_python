from matrix_types import Matrix
from time import now
from algorithm import vectorize
import benchmark

alias nelts = simdwidthof[DType.float32]() * 2
alias n = 10

fn kernel_floyd_warshall(path: Matrix):
    for k in range(n):
        for i in range(n):
           @parameter 
           fn calc[nelts: Int](j: Int):
                if (path[i,j] > path[i,k] + path[k,j]):
                        path.store[nelts](i,j, path.load[nelts](i,k) + path.load[nelts](k,j))
           vectorize[calc, nelts, size = n]()



@always_inline
fn benchmark_floyd_vector() -> object:
    var res = 0
    for i in range(10):
        var path = Matrix[n,n]().rand()

        var prev = now()
        kernel_floyd_warshall(path)
        var curr = now()
        res += curr - prev

        path.data.free()
    
    res = res // 10

    return res