from timeit import timeit
from t2mm import kernel_2mm, init_2mm
import numpy as np

def benchmark_python():
    A, B, C, D, alpha, beta, tmp = init_2mm(10,10,10,10)
    secs = timeit(lambda: kernel_2mm(A, B, C, D, alpha, beta, tmp, 10, 10, 10, 10), number=2)
    print(f"Test took {secs} seconds")


if __name__ == "__main__":
    benchmark_python()