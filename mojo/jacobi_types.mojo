from time import now
from matrix_types import Matrix
import benchmark

alias tsteps = 10
alias n = 10

fn kernel_jacobi_2d_types(tsteps: Int, n: Int, A: Matrix, B: Matrix):
    for t in range(tsteps):
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                B[i,j] = 0.2 * (A[i,j] + A[i,j-1] + A[i,j+1] + A[i+1,j] + A[i-1,j])
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                A[i,j] = B[i,j]


fn benchmark_jacobi_types() -> object:
    var res = 0
    for i in range(10):
        var A = Matrix[n,n]().rand()
        var B = Matrix[n,n]().rand()

        var prev = now()
        kernel_jacobi_2d_types(tsteps, n, A, B)
        var curr = now()
        res += curr - prev

        A.data.free()
        B.data.free()

    res = res // 10
    return res
