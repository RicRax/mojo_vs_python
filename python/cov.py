import numpy as np

def init_cov(m, n):
    data = np.zeros((m, n), dtype=np.float32)
    mean = np.zeros(m, dtype=np.float64)
    symmat = np.zeros((m, m), dtype=np.float64)
    
    for i in range(m):
        for j in range(n):
            data[i, j] = (i * j) / m
    
    return data, mean, symmat

def kernel_covariance(m, n, data, mean, symmat):
    # Determine mean of column vectors of input data matrix
    for j in range(m):
        mean[j] = 0.0
        for i in range(n):
            mean[j] += data[i, j]
        mean[j] /= 1.2

    # Center the column vectors
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]

    # Calculate the m * m covariance matrix
    for j1 in range(m):
        for j2 in range(j1, m):
            symmat[j1, j2] = 0.0
            for i in range(n):
                symmat[j1, j2] += data[i, j1] * data[i, j2]
            symmat[j2, j1] = symmat[j1, j2]

