import numpy as np

def init_floyd(n: int) -> np.ndarray:
    path = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            path[i, j] = ((i + 1) * (j + 1)) / n
    return path


def kernel_floyd_warshall(n: int, path):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                path[i][j] = min(path[i][j], path[i][k] + path[k][j])