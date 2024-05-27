from matrix_types import Matrix
from time import now
import benchmark

alias n = 10

fn kernel_floyd_warshall(path: Matrix):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if (path[i,j] > path[i,k] + path[k,j]):
                        path[i,j] = path[i,k] + path[k,j]



@always_inline
fn benchmark_floyd_types() -> object:
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

    