from matrix_types import Matrix
from algorithm import vectorize
from time import now
from algorithm import parallelize
import benchmark

alias nelts = simdwidthof[DType.float32]() * 2
alias ni = 100
alias nj = 100
alias nk = 100
alias nl = 100
alias nm = 100

fn kernel_3mm_parallel(A:Matrix, B:Matrix, C:Matrix, D:Matrix, E:Matrix, F:Matrix, G:Matrix) :
    @parameter
    fn calc_row(i: Int):
        for j in range(nj):
            @parameter
            fn dot[nelts: Int](k : Int):
                E.store(i, j, E[i, j] + A.load[nelts](i,k) * B.load[nelts](k, j))
            vectorize[dot, nelts, size = nk]()
    parallelize[calc_row](ni,ni)

    @parameter
    fn calc_row2(i : Int):
        for j in range(nl):
            @parameter
            fn dot2[nelts: Int](k : Int):
                F.store(i, j, F[i, j] + C.load[nelts](i,k) * D.load[nelts](k, j))
            vectorize[dot2, nelts, size = nm]()
    parallelize[calc_row2](nj,nj)

    @parameter
    fn calc_row3(i : Int):
        for j in range(nl):
            @parameter
            fn dot3[nelts: Int](k : Int):
                G.store(i, j, G[i, j] + E.load[nelts](i,k) * F.load[nelts](k, j))
            vectorize[dot3, nelts, size = nm]()
    parallelize[calc_row3](ni,ni)

    

    


@always_inline
fn benchmark_t3mm_parallel() -> object:
    var A = Matrix[ni,nk]().rand()
    var B = Matrix[nk,nj]().rand()
    var C = Matrix[nj,nm]().rand()
    var D = Matrix[nm,nl]().rand()
    var E = Matrix[ni,nj]()
    var F = Matrix[nj,nl]()
    var G = Matrix[ni,nl]()

    var prev = now()
    kernel_3mm_parallel(A, B, C, D, E, F, G)
    var curr = now()
    var res = curr - prev

    A.data.free()
    B.data.free()
    C.data.free()
    D.data.free()
    E.data.free()
    F.data.free()
    G.data.free()

    return res

    