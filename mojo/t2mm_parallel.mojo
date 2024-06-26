from matrix_types import Matrix
from algorithm import vectorize
from time import now
from algorithm import parallelize
import benchmark

alias nelts = simdwidthof[DType.float32]() * 2
alias ni = 800
alias nj = 800
alias nk = 800
alias nl = 800
alias nm = 800

fn kernel_2mm_parallel(A:Matrix, B:Matrix, C:Matrix, D:Matrix, alpha:Int, beta:Int, tmp:Matrix) :
    @parameter
    fn calc_row(i: Int):
        for j in range(nj):
            tmp[i,j] = 0
            @parameter
            fn dot[nelts: Int](k : Int):
                tmp.store(i, j, tmp[i, j] + alpha * A.load[nelts](i,k) * B.load[nelts](k, j))
            vectorize[dot, nelts, size = nk]()
    parallelize[calc_row](ni,ni)

    @parameter
    fn calc_row2(i : Int):
        for j in range(nl):
            D[i,j] *= beta
            @parameter
            fn dot2[nelts: Int](k : Int):
                D.store(i, j, D[i, j] + tmp.load[nelts](i,k) * C.load[nelts](k, j))
            vectorize[dot2, nelts, size = nj]()
    parallelize[calc_row2](ni,ni)


@always_inline
fn benchmark_t2mm_parallel() -> Float32:
    var res = 0
    for i in range(10):
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
        kernel_2mm_parallel(A, B, C, D, alpha, beta, tmp)
        var curr = now()
        res += curr - prev

        A.data.free()
        B.data.free()
        C.data.free()
        D.data.free()
        tmp.data.free()

    res = res // 10

    return res
