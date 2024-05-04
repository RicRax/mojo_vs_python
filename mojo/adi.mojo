from matrix import matrix_init
import benchmark


def kernel_adi(tsteps: Int, n: Int, X, A, B):
    for t in range(tsteps):
        for i1 in range(n):
            for i2 in range(1, n):
                X[i1][i2] = X[i1][i2] - X[i1][i2-1] * A[i1][i2] / B[i1][i2-1]
                B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2-1]

        for i1 in range(n):
            X[i1][n-1] = X[i1][n-1] / B[i1][ n-1]

        for i1 in range(n):
            for i2 in range(n-2, -1, -1):
                X[i1][i2] = (X[i1][i2+1] - X[i1][i2+1] * A[i1][i2+1]) / B[i1][i2]

        for i1 in range(1, n):
            for i2 in range(n):
                X[i1][i2] = X[i1][i2] - X[i1-1][i2] * A[i1][i2] / B[i1-1][i2]
                B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1-1][i2]

        for i2 in range(n):
            X[n-1][i2] = X[n-1][i2] / B[n-1][i2]

        for i1 in range(n-2, -1, -1):
            for i2 in range(n):
                X[i1][i2] = (X[i1+1][i2] - X[i1+1][i2] * A[i1+1][i2]) / B[i1][i2]



def benchmark_adi(tsteps: Int, n: Int):
    A = matrix_init(n, n)
    B = matrix_init(n, n)
    X = matrix_init(n, n)

    for i in range(n):
        a_row = object([])
        for j in range(n):
            a_row.append((i * (j + 1)) / n)
        A.append(a_row)
    
    for i in range(n):
        b_row = object([])
        for j in range(n):
            b_row.append((i * (j + 2)) / n)
        B.append(b_row)

    for i in range(n):
        x_row = object([])
        for j in range(n):
            b_row.append((i * (j + 3)) / n)
        X.append(x_row)

    @parameter
    fn test_fn():
        try:
            _ = kernel_adi(tsteps, n, X, A, B)
        except:
            pass

    var secs = benchmark.run[test_fn](max_runtime_secs=0.5).mean()
    _ = (A, B, X)
    return secs