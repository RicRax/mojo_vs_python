from timeit import timeit
from t2mm import kernel_2mm, init_2mm
from t3mm import kernel_3mm, init_3mm
from atax import kernel_atax, init_atax
from fdtd import init_fdtd, kernel_fdtd_2d
from jacobi import init_jacobi, kernel_jacobi_2d_imper
from floyd import init_floyd, kernel_floyd_warshall
from reg import init_reg, kernel_reg_detect
from cov import init_cov, kernel_covariance

def benchmark_python():
    A, B, C, D, alpha, beta, tmp = init_2mm(10,10,10,10)
    secs = timeit(lambda: kernel_2mm(A, B, C, D, alpha, beta, tmp, 10, 10, 10, 10), number=10)
    print(f"2mm took {secs} seconds")

    A, B, C, D, E, F, G = init_3mm(10,10,10,10,10)
    secs = timeit(lambda: kernel_3mm(10,10,10,10,10, A, B, C, D, E, F, G), number=10)
    print(f"3mm took {secs} seconds")

    A, x, y, tmp = init_atax(10,10)
    secs = timeit(lambda: kernel_atax(10,10, A, x, y, tmp), number=10)
    print(f"atax took {secs} seconds")

    ex, ey, hz, _fict_ = init_fdtd(10,10,10)
    secs = timeit(lambda: kernel_fdtd_2d(10,10,10,ex,ey,hz,_fict_), number=10)
    print(f"fdtd took {secs} seconds")

    A, B = init_jacobi(10)
    secs = timeit(lambda: kernel_jacobi_2d_imper(10, 10, A, B), number=10)
    print(f"jacobi took {secs} seconds")

    path = init_floyd(10)
    secs = timeit(lambda: kernel_floyd_warshall(10, path), number=10)
    print(f"floyd took {secs} seconds")

    sum_tang, mean, path, diff, sum_diff = init_reg(10, 10)
    secs = timeit(lambda: kernel_reg_detect(10, 10, 10, sum_tang, mean, path, diff, sum_diff), number=10)
    print(f"reg took {secs} seconds")

    data, mean, symmat = init_cov(10, 10)
    secs = timeit(lambda: kernel_covariance(10, 10, data, mean, symmat), number=10)
    print(f"covariance took {secs} seconds")

if __name__ == "__main__":
    benchmark_python()