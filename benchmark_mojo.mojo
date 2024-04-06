from t2mm import benchmark_t2mm
from t3mm import benchmark_t3mm


def main():
    t2mm_secs = benchmark_t2mm(10,10,10,10)
    print("2mm took", t2mm_secs, "seconds")

    t3mm_secs = benchmark_t3mm(10,10,10,10,10)
    print("3mm took", t3mm_secs, "seconds")