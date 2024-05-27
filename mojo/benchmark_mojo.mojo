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

def main():
    """
    t2mm_secs = benchmark_t2mm(100,100,100,100)
    print("2mm took", t2mm_secs, "seconds without types")

    t2mm_typed_secs = benchmark_2mm_types()
    print("2mm took", t2mm_typed_secs, "seconds with types")

    t2mm_vector_secs = benchmark_t2mm_vector()
    print("2mm vectorized took", t2mm_vector_secs, "seconds")

    #FASTER ONLY WITH LARGE SIZES
    t2mm_parallel_secs = benchmark_t2mm_parallel()
    print("2mm parallel took", t2mm_parallel_secs, "seconds")

    t3mm_secs = benchmark_t3mm(100,100,100,100,100)
    print("3mm took", t3mm_secs, "seconds without types")

    t3mm_typed_secs = benchmark_t3mm_typed()
    print("3mm took", t3mm_typed_secs, "seconds with types")

    t3mm_vector_secs = benchmark_t3mm_vector()
    print("3mm vectorized took", t3mm_vector_secs, "seconds")

    #FASTER ONLY WITH LARGE SIZES
    t3mm_parallel_secs = benchmark_t3mm_parallel()
    print("3mm parallel took", t3mm_parallel_secs, "seconds")

    fdtd_secs = benchmark_fdtd(10,10,10)
    print("fdtd took", fdtd_secs, "seconds")

    fdtd_typed_secs = benchmark_fdtd_types()
    print("fdtd took", fdtd_typed_secs, "seconds with types")

    fdtd_vector_secs = benchmark_fdtd_vector()
    print("fdtd vectorized took", fdtd_vector_secs, "seconds")

    jacobi_secs = benchmark_jacobi(10)
    print("jacobi took", jacobi_secs, "seconds")

    jacobi_typed_secs = benchmark_jacobi_types()
    print("jacobi took", jacobi_typed_secs, "seconds with types")

    jacobi_vector_secs = benchmark_jacobi_vector()
    print("jacobi vectorized took", jacobi_vector_secs, "seconds")

    floyd_secs = benchmark_floyd(10)
    print("floyd took", floyd_secs, "seconds")

    floyd_typed_secs = benchmark_floyd_types()
    print("floyd took", floyd_typed_secs, "seconds with types")

    floyd_vector_secs = benchmark_floyd_vector()
    print("floyd vectorized took", floyd_vector_secs, "seconds")

    floyd_parallel_secs = benchmark_floyd_parallel()
    print("floyd parallel took", floyd_parallel_secs, "seconds")
    """

    cov_secs = benchmark_cov(10, 10)
    print("covariance took", cov_secs, "seconds")