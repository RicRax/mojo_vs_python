from time import now
from algorithm import vectorize
from matrix_types import Matrix
import benchmark

alias nelts = simdwidthof[DType.float32]() * 2
alias tsteps = 500
alias n = 1300

fn kernel_jacobi_2d_vector(A: Matrix, B: Matrix):
    for t in range(tsteps):
        for i in range(1, n - 1):
            @parameter
            fn calc[nelts: Int](j: Int):
                if (j != 0):
                    B.store(i, j, 0.2 * (A.load[nelts](i, j) + A.load[nelts](i, j-1) + A.load[nelts](i, j+1) + A.load[nelts](i+1, j) + A.load[nelts](i-1, j)))
            vectorize[calc, nelts, size = n - 1]()

        for i in range(1, n - 1):
            @parameter
            fn calc2[nelts: Int](j: Int):
                if (j != 0):
                    A.store(i, j, B.load[nelts](i, j))
            vectorize[calc2, nelts, size = n - 1]()

fn benchmark_jacobi_vector() -> Float32:
    var res = 0
    for i in range(10):
        var A = Matrix[n, n]().rand()
        var B = Matrix[n, n]().rand()

        var prev = now()
        kernel_jacobi_2d_vector(A, B)
        var curr = now()
        res += curr - prev

        A.data.free()
        B.data.free()

    res = res // 10
    return res

