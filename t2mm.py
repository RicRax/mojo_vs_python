from timeit import timeit
import numpy as np

def kernel_2mm(ni, nj, nk, nl):
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


