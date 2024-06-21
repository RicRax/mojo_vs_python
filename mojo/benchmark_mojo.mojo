from python import Python
from t2mm import benchmark_t2mm
from t2mm_types import benchmark_2mm_types
from t2mm_vector import benchmark_t2mm_vector
from t2mm_parallel import benchmark_t2mm_parallel
from t3mm import benchmark_t3mm
from t3mm_types import benchmark_t3mm_typed
from t3mm_vector import benchmark_t3mm_vector
from t3mm_parallel import benchmark_t3mm_parallel
from atax import benchmark_atax
from fdtd import benchmark_fdtd
from fdtd_types import benchmark_fdtd_types
from fdtd_vector import benchmark_fdtd_vector
from jacobi import benchmark_jacobi
from jacobi_types import benchmark_jacobi_types
from jacobi_vector import benchmark_jacobi_vector
from floyd import benchmark_floyd
from floyd_types import benchmark_floyd_types
from floyd_vector import benchmark_floyd_vector
from floyd_parallel import benchmark_floyd_parallel
from cov import benchmark_cov
from cov_types import benchmark_cov_types
from cov_vector import benchmark_cov_vector
from cov_parallel import benchmark_cov_parallel

def update_results_csv(name: String, base: Float32, types: Float32, vector: Float32, parallel: Float32):
    Python.add_to_path(".")
    var save_results = Python.import_module("save_results")

    save_results.update_results(name, base /1000000000, types/1000000000, vector/1000000000, parallel/1000000000)

def run_benchmark_2mm(size: Int):
    python_secs = benchmark_t2mm(size, size, size, size)
    types_secs = benchmark_2mm_types()
    vector_secs = benchmark_t2mm_vector()
    parallel_secs = benchmark_t2mm_parallel()
    print(python_secs)
    print(types_secs)
    print(vector_secs)
    print(parallel_secs)
    update_results_csv("2mm", python_secs, types_secs, vector_secs, parallel_secs)

def run_benchmark_3mm(size: Int):
    python_secs = benchmark_t3mm(size, size, size, size, size)
    types_secs = benchmark_t3mm_typed()
    vector_secs = benchmark_t3mm_vector()
    parallel_secs = benchmark_t3mm_parallel()
    update_results_csv("3mm", python_secs, types_secs, vector_secs, parallel_secs)

def run_benchmark_fdtd(size1: Int, size2: Int, size3: Int):
    python_secs = benchmark_fdtd(size1, size2, size3)
    types_secs = benchmark_fdtd_types()
    vector_secs = benchmark_fdtd_vector()

    update_results_csv("fdtd", python_secs, types_secs, vector_secs, 0)

def run_benchmark_jacobi(size1: Int):
    python_secs = benchmark_jacobi(size1, size2)
    types_secs = benchmark_jacobi_types()
    vector_secs = benchmark_jacobi_vector()

    update_results_csv("jacobi", python_secs, types_secs, vector_secs, 0)

def run_benchmark_floyd(size: Int):
    python_secs = benchmark_floyd(size)
    types_secs = benchmark_floyd_types()
    vector_secs = benchmark_floyd_vector()
    parallel_secs = benchmark_floyd_parallel()
    update_results_csv("floyd", python_secs, types_secs, vector_secs, parallel_secs)

def run_benchmark_covariance(size1: Int, size2: Int):
    python_secs = benchmark_cov(size1, size2)
    types_secs = benchmark_cov_types()
    vector_secs = benchmark_cov_vector()
    parallel_secs = benchmark_cov_parallel()
    update_results_csv("covariance", python_secs, types_secs, vector_secs, parallel_secs)

def run_all_benchmarks():
    run_benchmark_2mm(800)
    run_benchmark_3mm(800)
    run_benchmark_fdtd(500, 200, 240)
    run_benchmark_jacobi(1300)
    run_benchmark_floyd(2800)
    run_benchmark_covariance(1024, 1024)

def main():
    run_all_benchmarks()


