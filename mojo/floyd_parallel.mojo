from time import now
from matrix_types import Matrix
from algorithm import parallelize, vectorize
import benchmark

alias n = 2800
alias nelts = simdwidthof[DType.float32]() * 2

fn kernel_floyd_warshall_parallel(path: Matrix):
    for k in range(n):
        @parameter
        fn calc_row(i: Int):
            @parameter
            fn calc[nelts: Int](j: Int):
                if (path[i,j] > path[i,k] + path[k,j]):
                        path.store[nelts](i,j, path.load[nelts](i,k) + path.load[nelts](k,j))
            vectorize[calc, nelts, size = n]()
        parallelize[calc_row](n, 8)

@always_inline
fn benchmark_floyd_parallel() -> Float32:
    var res = 0
    for _ in range(100):
        var path = Matrix[n, n]().rand()


        var prev = now()
        kernel_floyd_warshall_parallel(path)
        var curr = now()
        res += curr - prev

        path.data.free()
    
    res = res // 10
    return res
