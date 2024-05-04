from random import rand, random_float64
import benchmark

fn matrix_getitem(self: object, i: object) raises -> object:
    return self.value[i]


fn matrix_setitem(self: object, i: object, value: object) raises -> object:
    self.value[i] = value
    return None


fn matrix_append(self: object, value: object) raises -> object:
    self.value.append(value)
    return None


fn matrix_init(rows: Int, cols: Int) raises -> object:
    var value = object([])
    return object(
        Attr("value", value), Attr("__getitem__", matrix_getitem), Attr("__setitem__", matrix_setitem),
        Attr("rows", rows), Attr("cols", cols), Attr("append", matrix_append),
    )



alias type = DType.float32

struct Matrix[rows: Int, cols: Int]:
    var data: DTypePointer[type]

    # Initialize zeroeing all values
    fn __init__(inout self):
        self.data = DTypePointer[type].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, data: DTypePointer[type]):
        self.data = data

    # Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[type].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.store[width=nelts](y * self.cols + x, val)

def kernel_3mm(ni, nj, nk, nl, nm, A, B, C, D, E, F, G):
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                E[i][j] += A[i][k] * B[k][j]                


    for i in range(nj):
        for j in range(nl):
            for k in range(nm):
                F[i][j] += C[i][k] * D[k][j]

    for i in range(ni):
        for j in range(nl):
            for k in range(nj):
                G[i][j] += E[i][k] * F[k][j]




def benchmark_t3mm(ni: Int ,nj: Int, nk: Int,nl: Int, nm: Int):
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

    @parameter
    fn test_fn():
        try:
            _ = kernel_3mm(ni, nj, nk, nl, nm, A, B, C, D, E, F, G )
        except:
            print("error occurred")

    var secs = benchmark.run[test_fn](max_runtime_secs=0.5).mean()
    _ = (A, B, C, D, E, F, G)
    return secs



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


fn main():
    var t3mm_typed_secs = benchmark_t3mm_typed[kernel_3mm_types]()
    print("3mm took", t3mm_typed_secs, "seconds with types")

    try:
        var t3mm_secs = benchmark_t3mm(10,10,10,10,10)
        print("3mm took", t3mm_secs, "seconds without types")
    except:
        print("error occurred")

    