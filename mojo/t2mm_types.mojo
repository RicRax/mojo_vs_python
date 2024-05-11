from matrix_types import Matrix
from time import now
import benchmark

fn kernel_2mm_types(A:Matrix, B:Matrix, C:Matrix, D:Matrix, ni:Int, nj:Int, nk:Int, nl:Int, alpha:Int, beta:Int, tmp:Matrix):
    for i in range(ni):
        for j in range(nj):
            tmp[i,j] = 0
            for k in range(nk):
                tmp[i,j] += alpha * A[i,k] * B[k,j]

    for i in range(ni):
        for j in range(nl):
            D[i,j] *= beta
            for k in range(nj):
                D[i,j] += tmp[i,k] * C[k,j]



alias ni = 100
alias nj = 100
alias nk = 100
alias nl = 100
alias nm = 100

@always_inline
fn benchmark_2mm_types() -> object:

    var A = Matrix[ni,nk]()
    var B = Matrix[nk,nj]()
    var C = Matrix[nl,nj]()
    var D = Matrix[ni,nl]()
    var tmp = Matrix[ni,nj]()

    for i in range(ni):
        for j in range(nk):
            A[i,j] = (i * j) / ni
            
    for i in range(nk):
        for j in range(nj):
            B[i,j] = (i * (j + 1)) / nj
            
    for i in range(nl):
        for j in range(nj):
            C[i,j] = (i * (j + 3)) / nl
            
    for i in range(ni):
        for j in range(nl):
            D[i,j] = (i * (j + 2)) / nk

    var alpha : Int = 32412 
    var beta : Int = 2123 

    var prev = now()
    kernel_2mm_types(A,B,C,D,ni,nj,nk,nl,alpha,beta,tmp)
    var curr = now()
    var res = curr - prev

    A.data.free()
    B.data.free()
    C.data.free()
    D.data.free()

    return res
