from t2mm import benchmark_matmul_untyped


def main():
    t2mm_secs = benchmark_matmul_untyped(10,10,10,10)
    print("2mm took", t2mm_secs, "seconds")
