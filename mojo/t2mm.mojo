from matrix import matrix_init
import benchmark


def kernel_2mm(A, B, C, D, ni, nj, nk, nl, alpha, beta, tmp):
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

def benchmark_t2mm(ni: Int ,nj: Int, nk: Int,nl: Int):
    A = matrix_init(ni,nk)
    B = matrix_init(nk,nj)
    C = matrix_init(nl,nj)
    D = matrix_init(ni,nl)
    tmp = matrix_init(ni,nj)

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

    for i in range(nl):
        c_row = object([])
        for j in range(nj):
            c_row.append((i * (j + 3)) / nl)
        C.append(c_row)

    for i in range(ni):
        d_row = object([])
        for j in range(nl):
            d_row.append((i * (j + 2)) / nk)
        D.append(d_row)

    for i in range(ni):
        tmp_row = object([])
        for j in range(nj):
            tmp_row.append(0.0)
        tmp.append(tmp_row)

    alpha = 32412
    beta = 2123

    @parameter
    fn test_fn():
        try:
            _ = kernel_2mm(A, B, C, D, ni, nj, nk, nl, alpha, beta, tmp)
        except:
            pass

    var secs = benchmark.run[test_fn](max_runtime_secs=0.5).mean()
    _ = (A, B, C, D)
    return secs
    