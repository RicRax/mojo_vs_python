from matrix_types import Matrix
import benchmark

fn kernel_3mm_types(ni: Int, nj: Int, nk: Int, nl: Int, nm: Int, A:Matrix, B:Matrix, C:Matrix, D:Matrix, E:Matrix, F:Matrix, G:Matrix) raises:
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
    

    

alias ni = 10
alias nj = 10
alias nk = 10
alias nl = 10
alias nm = 10

@always_inline
fn benchmark_t3mm_typed[
    func: fn (Int, Int, Int, Int, Int, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix) raises -> None]() -> object:
    var A = Matrix[ni,nk]()
    var B = Matrix[nk,nj]()
    var C = Matrix[nj,nm]()
    var D = Matrix[nm,nl]()
    var E = Matrix[ni,nj]()
    var F = Matrix[nj,nl]()
    var G = Matrix[ni,nl]()

    for i in range(ni):
        for j in range(nk):
            A[i,j] = (i * j) / ni
            

    for i in range(nk):
        for j in range(nj):
            B[i,j] = (i * (j + 1)) / nj
            
    for i in range(nj):
        for j in range(nm):
            C[i,j] = (i * (j + 3)) / nl
            
    for i in range(nm):
        for j in range(nl):
            D[i,j] = (i * (j + 2)) / nk

    @always_inline
    @parameter
    fn test_fn():
        try:
            _ = func(ni, nj, nk, nl, nm, A, B, C, D, E, F, G)
        except:
            print("error occurred")

    var secs = benchmark.run[test_fn](max_runtime_secs=0.5).mean()

    A.data.free()
    B.data.free()
    C.data.free()
    D.data.free()
    E.data.free()
    F.data.free()
    G.data.free()

    return secs

    