from matrix import matrix_init
from time import now
import benchmark


def kernel_jacobi_2d_imper(tsteps: Int, n: Int, A, B):
    for t in range(tsteps):
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                A[i, j] = B[i, j]


def benchmark_jacobi(n: Int) -> Float32:
    var res = 0
    for i in range(10):
        A = matrix_init(n,n)
        B = matrix_init(n,n)

        for i in range(n):
            a_row = object([])
            b_row = object([])
            for j in range(n):
                a_row.append(((i * (j + 2)) + 2) / n)
                b_row.append(((i * (j + 3)) + 3) / n)
            A.append(a_row)
            B.append(b_row)

        var prev = now()
        kernel_jacobi_2d_imper(500,n,A,B)
        var curr = now()
        res += curr - prev

        _ = (A, B)
    
    res = res // 10
    return res

    
