from matrix import matrix_init
import benchmark
from collections.vector import InlinedFixedVector
from python import Python

def kernel_atax(nx, ny, A, x, y, tmp):
    for i in range(ny):
        y[0][i] = 0 

    for i in range(nx):
        for j in range(ny):
            tmp[0][i] = tmp[0][i] + A[i][j] * x[0][j]
        for j in range(ny):
            y[0][j] = y[0][j] + A[i][j] * tmp[0][i] 

    return y

def benchmark_atax(nx: Int, ny: Int):
    x_vector = matrix_init(ny, 1)
    y_vector = matrix_init(ny, 1)
    A = matrix_init(nx, ny)
    tmp = matrix_init(nx, 1)
    #tmp = InlinedFixedVector[FloatLiteral](nx)
    #math = Python.import_module("math")

    #pi = math.pi
    #pi = pi.to_float64()

    x_row = object([])
    for i in range(ny):
        x_row.append(i * 3.14)
    x_vector.append(x_row)

    y_row = object([])
    for i in range(ny):
        y_row.append(0.0)
    y_vector.append(y_row)

    for i in range(nx):
        a_row = object([])
        for j in range(ny):
            a_row.append((i * (j + 1)) / nx)
        A.append(a_row)

    tmp_row = object([])
    for i in range(ny):
        tmp_row.append(0.0)
    tmp.append(tmp_row)

    @parameter
    fn test_fn():
        try:
            _ = kernel_atax(nx, ny, A, x_vector, y_vector, tmp)
        except:
            pass

    var secs = benchmark.run[test_fn](max_runtime_secs=0.5).mean()
    _ = (A, x_vector, y_vector, tmp)
    return secs
    