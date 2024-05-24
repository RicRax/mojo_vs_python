import numpy as np

def init_jacobi(n):
    A = np.zeros((n, n), dtype=np.float64)
    B = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            A[i, j] = (i * (j + 2) + 2) / n
            B[i, j] = (i * (j + 3) + 3) / n

    return A, B

def kernel_jacobi_2d_imper(tsteps, n, A, B):
    for t in range(tsteps):
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                A[i, j] = B[i, j]