from matrix_types import Matrix
from matrix import matrix_init
from time import now
from algorithm import vectorize

alias m = 10
alias n = 10
alias nelts = simdwidthof[DType.float32]() * 2  

fn kernel_covariance(data: Matrix, mean: Matrix, symmat: Matrix):
    # Determine mean of column vectors of input data matrix
    for j in range(m):
        mean[0, j] = 0.0
        for i in range(n):
            mean[0, j] += data[i, j]
        mean[0, j] /= 1.2

    # Center the column vectors
    for i in range(n):
        @parameter
        fn center_col[nelts: Int](j: Int):
            data.store(i, j, data.load[nelts](i, j) - mean.load[nelts](0, j))
        vectorize[center_col, nelts, size=m]()

    # Calculate the m * m covariance matrix
    for j1 in range(m):
        for j2 in range(j1, m):
            symmat[j1, j2] = 0.0
            @parameter
            fn calc_cov[nelts: Int](i: Int):
                symmat.store(j1,j2, symmat[j1,j2] + data.load[nelts](i, j1) * data.load[nelts](i, j2))
            vectorize[calc_cov, nelts, size=n]()
            symmat[j2, j1] = symmat[j1, j2]

fn benchmark_cov_vector() -> object:
    var res = 0
    for i in range(10):
        var data = Matrix[m, n]().rand()
        var mean = Matrix[1, n]()
        var symmat = Matrix[m, n]()

        var prev = now()
        kernel_covariance(data, mean, symmat)
        var curr = now()
        res += curr - prev

        data.data.free()
        mean.data.free()
        symmat.data.free()

    res = res // 10

    return res
