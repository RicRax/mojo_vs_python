from matrix import matrix_init
from matrix_types import Matrix
from time import now
import benchmark

def kernel_3mm(ni, nj, nk, nl, nm, A, B, C, D, E, F, G):
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                E[i,j] += A[i,k] * B[k,j]                

    for i in range(nj):
        for j in range(nl):
            for k in range(nm):
                F[i,j] += C[i,k] * D[k,j]

    for i in range(ni):
        for j in range(nl):
            for k in range(nj):
                G[i,j] += E[i,k] * F[k,j]





def benchmark_t3mm(ni: Int ,nj: Int, nk: Int,nl: Int, nm: Int) -> Float32:
    A = matrix_init(ni, nk)
    B = matrix_init(nk, nj)
    C = matrix_init(nj, nm)
    D = matrix_init(nm, nl)
    E = matrix_init(ni, nj)
    F = matrix_init(nj, nl)
    G = matrix_init(ni, nl)

    for i in range(ni):
        a_row = object([])
        for j in range(nk):
            a_row.append((i * j) / ni)
        A.append(a_row)

    
    for i in range(nk):
        b_row = object([])
        for j in range(nj):
            b_row.append((i * (j + 1)) / nj)
        B.append(b_row)

    for i in range(nj):
        c_row = object([])
        for j in range(nm):
            c_row.append((i * (j + 3)) / nl)
        C.append(c_row)

    for i in range(nm):
        d_row = object([])
        for j in range(nl):
            d_row.append((i * (j + 2)) / nk)
        D.append(d_row)

    for i in range(ni):
        e_row = object([])
        for j in range(nj):
            e_row.append(0.0)
        E.append(e_row)

    for i in range(nj):
        f_row = object([])
        for j in range(nl):
            f_row.append(0.0)
        F.append(f_row)

    for i in range(ni):
        g_row = object([])
        for j in range(nl):
            g_row.append(0.0)
        G.append(g_row)

    prev = now()
    kernel_3mm(ni, nj, nk, nl, nm, A, B, C, D, E, F, G)
    curr = now()
    res = curr - prev
    
    return res
