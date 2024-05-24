from matrix_types import Matrix
from time import now
import benchmark

alias ni = 100
alias nj = 100
alias nk = 100
alias nl = 100
alias nm = 100

fn kernel_3mm_types(ni: Int, nj: Int, nk: Int, nl: Int, nm: Int, A:Matrix, B:Matrix, C:Matrix, D:Matrix, E:Matrix, F:Matrix, G:Matrix):
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



@always_inline
fn benchmark_t3mm_typed() -> object:

    var res = 0
    for i in range(10):
        var A = Matrix[ni,nk]().rand()
        var B = Matrix[nk,nj]().rand()
        var C = Matrix[nj,nm]().rand()
        var D = Matrix[nm,nl]().rand()
        var E = Matrix[ni,nj]().rand()
        var F = Matrix[nj,nl]().rand()
        var G = Matrix[ni,nl]().rand()

        var prev = now()
        kernel_3mm_types(ni, nj, nk, nl, nm, A, B, C, D, E, F, G)
        var curr = now()
        res += curr - prev

        A.data.free()
        B.data.free()
        C.data.free()
        D.data.free()
        E.data.free()
        F.data.free()
        G.data.free()
    
    res = res // 10

    return res

    