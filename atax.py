import math
import numpy as np

def init_atax(nx, ny):
    x_vector = np.zeros(ny)
    y_vector = np.zeros(ny)
    A_matrix = np.zeros((nx,ny))
    tmp = np.zeros(nx)

    for i in range(ny):
        x_vector[i] = i * math.pi

    for i in range(nx):
        for j in range(ny):
            A_matrix[i][j] = ((i * (j + 1)) / nx)

    return A_matrix, x_vector, y_vector, tmp

def kernel_atax(nx, ny, A, x, y, tmp):
    for i in range(ny):
        y[i] = 0

    for i in range(nx):
        for j in range(ny):
            tmp[i] = tmp[i] + A[i][j] * x[j]
        for j in range(ny):
            y[j] = y[j] + A[i][j] * tmp[i]

    return y


