from matrix import matrix_init
from time import now
from matrix_types import Matrix
import benchmark

def kernel_floyd_warshall(n, path):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if (path[i][j] > path[i][k] + path[k][j]):
                        path[i][j] = path[i][k] + path[k][j]


def benchmark_floyd(n: Int) -> Float32:
    var res = 0
    for i in range(10):
        path = matrix_init(n,n)

        for i in range(n):
            row = object([])
            for j in range(n):
                row.append(((i + 1) * (j + 1)) / n)
            path.append(row)

        var prev = now()
        kernel_floyd_warshall(n, path)
        var curr = now()
        res += curr - prev

        _ = (path)
    
    res = res // 10
    return res

    
