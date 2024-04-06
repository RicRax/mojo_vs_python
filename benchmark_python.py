from timeit import timeit
from t2mm import kernel_2mm, init_2mm
from t3mm import kernel_3mm, init_3mm
from atax import kernel_atax, init_atax

def benchmark_python():
    A, B, C, D, alpha, beta, tmp = init_2mm(10,10,10,10)
    secs = timeit(lambda: kernel_2mm(A, B, C, D, alpha, beta, tmp, 10, 10, 10, 10), number=2)
    print(f"2mm took {secs} seconds")

    A, B, C, D, E, F, G = init_3mm(10,10,10,10,10)
    secs = timeit(lambda: kernel_3mm(10,10,10,10,10, A, B, C, D, E, F, G), number=2)
    print(f"3mm took {secs} seconds")

    A, x, y, tmp = init_atax(10,10)
    secs = timeit(lambda: kernel_atax(10,10, A, x, y, tmp), number=2)
    print(f"atax took {secs} seconds")

if __name__ == "__main__":
    benchmark_python()