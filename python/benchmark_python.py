import os
import csv
from timeit import timeit
from t2mm import kernel_2mm, init_2mm, kernel_2mm_numba, kernel_2mm_numpy
from t3mm import kernel_3mm, init_3mm, kernel_3mm_numba, kernel_3mm_numpy
from fdtd import init_fdtd, kernel_fdtd_2d, kernel_fdtd_2d_numba, kernel_fdtd_2d_numpy
from jacobi import (
    init_jacobi,
    kernel_jacobi_2d_imper,
    kernel_jacobi_2d_imper_numba,
    kernel_jacobi_2d_imper_numpy,
)
from floyd import (
    init_floyd,
    kernel_floyd_warshall,
    kernel_floyd_warshall_numba,
    kernel_floyd_warshall_numpy,
)
from cov import (
    init_cov,
    kernel_covariance,
    kernel_covariance_numba,
    kernel_covariance_numpy,
)


def benchmark_2mm(size, iterations):
    A, B, C, D, alpha, beta, tmp = init_2mm(size, size, size, size)
    base_secs = timeit(
        lambda: kernel_2mm(A, B, C, D, alpha, beta, tmp, size, size, size, size),
        number=iterations,
    )

    A, B, C, D, alpha, beta, tmp = init_2mm(size, size, size, size)
    numba_secs = timeit(
        lambda: kernel_2mm_numba(A, B, C, D, alpha, beta, tmp, size, size, size, size),
        number=iterations,
    )

    A, B, C, D, alpha, beta, tmp = init_2mm(size, size, size, size)
    numpy_secs = timeit(
        lambda: kernel_2mm_numpy(A, B, C, D, alpha, beta, tmp), number=iterations
    )

    update_results_csv("2mm_full", base_secs, numba_secs, numpy_secs)


def benchmark_3mm(size, iterations):
    A, B, C, D, E, F, G = init_3mm(size, size, size, size, size)
    base_secs = timeit(
        lambda: kernel_3mm(size, size, size, size, size, A, B, C, D, E, F, G),
        number=iterations,
    )

    A, B, C, D, E, F, G = init_3mm(size, size, size, size, size)
    numba_secs = timeit(
        lambda: kernel_3mm_numba(size, size, size, size, size, A, B, C, D, E, F, G),
        number=iterations,
    )

    A, B, C, D, E, F, G = init_3mm(size, size, size, size, size)
    numpy_secs = timeit(
        lambda: kernel_3mm_numpy(A, B, C, D, E, F, G), number=iterations
    )

    update_results_csv("3mm_full", base_secs, numba_secs, numpy_secs)


def benchmark_fdtd(size1, size2, size3, iterations):
    ex, ey, hz, _fict_ = init_fdtd(size1, size2, size3)
    base_secs = timeit(
        lambda: kernel_fdtd_2d(size1, size2, size3, ex, ey, hz, _fict_),
        number=iterations,
    )

    ex, ey, hz, _fict_ = init_fdtd(size1, size2, size3)
    numba_secs = timeit(
        lambda: kernel_fdtd_2d_numba(size1, size2, size3, ex, ey, hz, _fict_),
        number=iterations,
    )

    ex, ey, hz, _fict_ = init_fdtd(size1, size2, size3)
    numpy_secs = timeit(
        lambda: kernel_fdtd_2d_numpy(size1, size2, size3, ex, ey, hz, _fict_),
        number=iterations,
    )

    update_results_csv("fdtd_full", base_secs, numba_secs, numpy_secs)


def benchmark_jacobi(size, iterations):
    A, B = init_jacobi(size)
    base_secs = timeit(
        lambda: kernel_jacobi_2d_imper(size, size, A, B), number=iterations
    )

    A, B = init_jacobi(size)
    numba_secs = timeit(
        lambda: kernel_jacobi_2d_imper_numba(size, size, A, B), number=iterations
    )

    A, B = init_jacobi(size)
    numpy_secs = timeit(
        lambda: kernel_jacobi_2d_imper_numpy(size, size, A, B), number=iterations
    )

    update_results_csv("jacobi_full", base_secs, numba_secs, numpy_secs)


def benchmark_floyd(size, iterations):
    path = init_floyd(size)
    base_secs = timeit(lambda: kernel_floyd_warshall(size, path), number=iterations)

    path = init_floyd(size)
    numba_secs = timeit(
        lambda: kernel_floyd_warshall_numba(size, path), number=iterations
    )

    path = init_floyd(size)
    numpy_secs = timeit(
        lambda: kernel_floyd_warshall_numpy(size, path), number=iterations
    )

    update_results_csv("floyd_full", base_secs, numba_secs, numpy_secs)


def benchmark_covariance(size1, size2, iterations):
    data, mean, symmat = init_cov(size1, size2)
    base_secs = timeit(
        lambda: kernel_covariance(size1, size2, data, mean, symmat), number=iterations
    )

    data, mean, symmat = init_cov(size1, size2)
    numba_secs = timeit(
        lambda: kernel_covariance_numba(size, size, data, mean, symmat),
        number=iterations,
    )

    data, mean, symmat = init_cov(size1, size2)
    numpy_secs = timeit(
        lambda: kernel_covariance_numpy(data, mean, symmat), number=iterations
    )

    update_results_csv("covariance_full", base_secs, numba_secs, numpy_secs)


def run_all_benchmarks():
    benchmark_2mm(800, 10)
    benchmark_3mm(800, 10)
    benchmark_fdtd(500, 200, 240, 10)
    benchmark_jacobi(1300, 10)
    benchmark_floyd(2800, 10)
    benchmark_covariance(1024, 1024, 10)


def update_results_csv(name, python, numba, numpy):
    file_exists = os.path.isfile("results.csv")

    with open("results.csv", mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header if the file does not exist
        if not file_exists:
            writer.writerow(["name", "python", "numba", "numpy"])

        # Write the data row
        writer.writerow([name, python, numba, numpy])


def print_results_csv():
    if not os.path.isfile("results.csv"):
        print("The file 'results.csv' does not exist.")
        return

    with open("results.csv", mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)


if __name__ == "__main__":
    run_all_benchmarks()
