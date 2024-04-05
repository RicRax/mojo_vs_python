from timeit import timeit
from t2mm import kernel_2mm
import numpy as np

def benchmark_python():
    secs = timeit(lambda: kernel_2mm(10, 10, 10, 10), number=2)
    print(f"Test took {secs} seconds")


if __name__ == "__main__":
    benchmark_python()