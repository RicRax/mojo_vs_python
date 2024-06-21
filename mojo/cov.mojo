from matrix import matrix_init
from time import now

def kernel_covariance(m: Int, n: Int, data, mean, symmat):
    # Determine mean of column vectors of input data matrix
    for j in range(m):
        mean[0][j] = 0.0
        for i in range(n):
            mean[0][j] += data[i, j]
        mean[0][j] /= 1.2

    # Center the column vectors
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[0][j]

    # Calculate the m * m covariance matrix
    for j1 in range(m):
        for j2 in range(j1, m):
            symmat[j1, j2] = 0.0
            for i in range(n):
                symmat[j1, j2] += data[i, j1] * data[i, j2]
            symmat[j2, j1] = symmat[j1, j2]


def benchmark_cov(m: Int, n:Int) -> Float32:
    var res = 0
    for i in range(10):
        data = matrix_init(m,n)
        mean = matrix_init(1,m)
        symmat = matrix_init(m,m)

        for i in range(m):
            row = object([])
            for j in range(n):
                row.append((i * j) / m)
            data.append(row)
    
        for i in range(m):
            row = object([])
            for j in range(n):
                row.append(0.0)
            symmat.append(row)

        row = []
        for j in range(n):
            row.append(0.0)
        mean.append(row)


        var prev = now()
        kernel_covariance(m, n, data, mean, symmat)
        var curr = now()
        res += curr - prev
        _ = (data, mean, symmat)

    res = res // 10

    return res
