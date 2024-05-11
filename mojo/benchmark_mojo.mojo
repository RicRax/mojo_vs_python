from t2mm import benchmark_t2mm
from t3mm import benchmark_t3mm
from t3mm_types import benchmark_t3mm_typed
from t3mm_vector import benchmark_t3mm_vector
from t3mm_parallel import benchmark_t3mm_parallel
from t2mm_types import benchmark_2mm_types
from atax import benchmark_atax
from fdtd import benchmark_fdtd


def main():
    t2mm_secs = benchmark_t2mm(100,100,100,100)
    print("2mm took", t2mm_secs, "seconds without types")

    t2mm_typed_secs = benchmark_2mm_types()
    print("2mm took", t2mm_typed_secs, "seconds with types")

    t3mm_vector_secs = benchmark_t3mm_vector()
    print("3mm vectorized took", t3mm_vector_secs, "seconds")

    t3mm_typed_secs = benchmark_t3mm_typed()
    print("3mm took", t3mm_typed_secs, "seconds with types")

    t3mm_secs = benchmark_t3mm(100,100,100,100,100)
    print("3mm took", t3mm_secs, "seconds without types")

    atax_secs = benchmark_atax(100,100)
    print("atax took", atax_secs, "seconds")

    fdtd_secs = benchmark_fdtd(10,10,10)
    print("fdtd took", fdtd_secs, "seconds")


    t3mm_parallel_secs = benchmark_t3mm_parallel()
    print("3mm parallel took", t3mm_parallel_secs, "seconds")