from t2mm import benchmark_t2mm
from t3mm import benchmark_t3mm
from t3mm_types import benchmark_t3mm_typed
from t3mm_types import kernel_3mm_types
from t2mm_types import benchmark_2mm_types
from t2mm_types import kernel_2mm_types
from atax import benchmark_atax
from adi import benchmark_adi


def main():
    t2mm_secs = benchmark_t2mm(10,10,10,10)
    print("2mm took", t2mm_secs, "seconds")

    t2mm_typed_secs = benchmark_2mm_types[kernel_2mm_types]()
    print("2mm took", t2mm_typed_secs, "seconds with types")

    t3mm_typed_secs = benchmark_t3mm_typed[kernel_3mm_types]()
    print("3mm took", t3mm_typed_secs, "seconds with types")

    t3mm_secs = benchmark_t3mm(10,10,10,10,10)
    print("3mm took", t3mm_secs, "seconds without types")

    atax_secs = benchmark_atax(10,10)
    print("atax took", atax_secs, "seconds")

    adi_secs = benchmark_atax(10,10)
    print("adi took", adi_secs, "seconds")