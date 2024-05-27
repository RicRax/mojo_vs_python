from matrix_types import Matrix
from matrix import matrix_init
from time import now

alias m = 10
alias n = 10

fn kernel_covariance(data: Matrix, mean: Matrix, symmat: Matrix):
    # Determine mean of column vectors of input data matrix
    for j in range(m):
        mean[0,j] = 0.0
        for i in range(n):
            mean[0,j] += data[i, j]
        mean[0,j] /= 1.2

    # Center the column vectors
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[0,j]

    # Calculate the m * m covariance matrix
    for j1 in range(m):
        for j2 in range(j1, m):
            symmat[j1, j2] = 0.0
            for i in range(n):
                symmat[j1, j2] += data[i, j1] * data[i, j2]
            symmat[j2, j1] = symmat[j1, j2]


fn benchmark_cov_types() -> object:
    var res = 0
    for i in range(10):
        var data = Matrix[m,n]().rand()
        var mean = Matrix[1,n]()
        var symmat = Matrix[m,n]()

        var prev = now()
        kernel_covariance(data, mean, symmat)
        var curr = now()
        res += curr - prev

        data.data.free()
        mean.data.free()
        symmat.data.free()

    res = res // 10

    return res