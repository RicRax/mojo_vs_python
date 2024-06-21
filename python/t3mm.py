from timeit import timeit
import numpy as np
from numba import jit


def init_3mm(ni, nj, nk, nl, nm):
    A = np.zeros((ni, nk))
    B = np.zeros((nk, nj))
    C = np.zeros((nj, nm))
    D = np.zeros((nm, nl))
    E = np.zeros((ni, nj))
    F = np.zeros((nj, nl))
    G = np.zeros((ni, nl))

    for i in range(ni):
        for j in range(nk):
            A[i][j] = (i * j) / ni

    for i in range(nk):
        for j in range(nj):
            B[i][j] = (i * (j + 1)) / nj

    for i in range(nj):
        for j in range(nm):
            C[i][j] = (i * (j + 3)) / nl

    for i in range(nm):
        for j in range(nl):
            D[i][j] = (i * (j + 2)) / nk

    return A, B, C, D, E, F, G


def kernel_3mm(ni, nj, nk, nl, nm, A, B, C, D, E, F, G):
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                E[i, j] += A[i, k] * B[k, j]

    for i in range(nj):
        for j in range(nl):
            for k in range(nm):
                F[i, j] += C[i, k] * D[k, j]

    for i in range(ni):
        for j in range(nl):
            for k in range(nj):
                G[i, j] += E[i, k] * F[k, j]
    return G


def kernel_3mm_numpy(A, B, C, D, E, F, G):
    # E := A * B
    E = np.dot(A, B)

    # F := C * D
    F = np.dot(C, D)

    # G := E * F
    G = np.dot(E, F)

    return G


@jit(nopython=True)
def kernel_3mm_numba(ni, nj, nk, nl, nm, A, B, C, D, E, F, G):
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                E[i, j] += A[i, k] * B[k, j]

    for i in range(nj):
        for j in range(nl):
            for k in range(nm):
                F[i, j] += C[i, k] * D[k, j]

    for i in range(ni):
        for j in range(nl):
            for k in range(nj):
                G[i, j] += E[i, k] * F[k, j]
    return G
