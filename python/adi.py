from timeit import timeit
import numpy as np


def init_adi(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    X = np.random.rand(n, n)

    return A, B, X




def kernel_adi(tsteps, n, X, A, B):
    for t in range(tsteps):
        for i1 in range(n):
            for i2 in range(1, n):
                X[i1, i2] = X[i1, i2] - X[i1, i2-1] * A[i1, i2] / B[i1, i2-1]
                B[i1, i2] = B[i1, i2] - A[i1, i2] * A[i1, i2] / B[i1, i2-1]

        for i1 in range(n):
            X[i1, n-1] = X[i1, n-1] / B[i1, n-1]

        for i1 in range(n):
            for i2 in range(n-2, -1, -1):
                X[i1, i2] = (X[i1, i2+1] - X[i1, i2+1] * A[i1, i2+1]) / B[i1, i2]

        for i1 in range(1, n):
            for i2 in range(n):
                X[i1, i2] = X[i1, i2] - X[i1-1, i2] * A[i1, i2] / B[i1-1, i2]
                B[i1, i2] = B[i1, i2] - A[i1, i2] * A[i1, i2] / B[i1-1, i2]

        for i2 in range(n):
            X[n-1, i2] = X[n-1, i2] / B[n-1, i2]

        for i1 in range(n-2, -1, -1):
            for i2 in range(n):
                X[i1, i2] = (X[i1+1, i2] - X[i1+1, i2] * A[i1+1, i2]) / B[i1, i2]

