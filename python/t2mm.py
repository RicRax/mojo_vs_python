from timeit import timeit
import numpy as np
from numba import jit


def init_2mm(ni, nj, nk, nl):
    alpha = 32412
    beta = 2123
    A = np.zeros((ni, nk))
    B = np.zeros((nk, nj))
    C = np.zeros((nl, nj))
    D = np.zeros((ni, nl))

    for i in range(ni):
        for j in range(nk):
            A[i][j] = (i * j) / ni

    for i in range(nk):
        for j in range(nj):
            B[i][j] = (i * (j + 1)) / nj

    for i in range(nl):
        for j in range(nj):
            C[i][j] = (i * (j + 3)) / nl

    for i in range(ni):
        for j in range(nl):
            D[i][j] = (i * (j + 2)) / nk

    tmp = np.zeros((ni, nj))

    return A, B, C, D, alpha, beta, tmp


def kernel_2mm(A, B, C, D, alpha, beta, tmp, ni, nj, nk, nl):
    # D := alpha*A*B*C + beta*D
    for i in range(ni):
        for j in range(nj):
            tmp[i][j] = 0
            for k in range(nk):
                tmp[i][j] += alpha * A[i][k] * B[k][j]

    for i in range(ni):
        for j in range(nl):
            D[i][j] *= beta
            for k in range(nj):
                D[i][j] += tmp[i][k] * C[k][j]

    return D


def kernel_2mm_numpy(A, B, C, D, alpha, beta, tmp):
    tmp = np.dot(A, B) * alpha
    D = beta * D + np.dot(tmp, C)

    return D


@jit(nopython=True)
def kernel_2mm_numba(A, B, C, D, alpha, beta, tmp, ni, nj, nk, nl):
    # D := alpha*A*B*C + beta*D
    for i in range(ni):
        for j in range(nj):
            tmp[i][j] = 0
            for k in range(nk):
                tmp[i][j] += alpha * A[i][k] * B[k][j]

    for i in range(ni):
        for j in range(nl):
            D[i][j] *= beta
            for k in range(nj):
                D[i][j] += tmp[i][k] * C[k][j]

    return D
