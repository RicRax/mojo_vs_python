import numpy as np

def init_reg(maxgrid, length):
    sum_tang = np.zeros((maxgrid, maxgrid), dtype=np.float64)
    mean = np.zeros((maxgrid, maxgrid), dtype=np.float64)
    path = np.zeros((maxgrid, maxgrid), dtype=np.float64)
    diff = np.zeros((maxgrid, maxgrid, length), dtype=np.float64)
    sum_diff = np.zeros((maxgrid, maxgrid, length), dtype=np.float64)

    for i in range(maxgrid):
        for j in range(maxgrid):
            sum_tang[i, j] = float((i + 1) * (j + 1))
            mean[i, j] = float(i - j) / maxgrid
            path[i, j] = float(i * (j - 1)) / maxgrid

    return sum_tang, mean, path,diff, sum_diff

def kernel_reg_detect(niter, maxgrid, length, sum_tang, mean, path, diff, sum_diff):
    for t in range(niter):
        for j in range(maxgrid):
            for i in range(j, maxgrid):
                for cnt in range(length):
                    diff[j, i, cnt] = sum_tang[j, i]

        for j in range(maxgrid):
            for i in range(j, maxgrid):
                sum_diff[j, i, 0] = diff[j, i, 0]
                for cnt in range(1, length):
                    sum_diff[j, i, cnt] = sum_diff[j, i, cnt - 1] + diff[j, i, cnt]
                mean[j, i] = sum_diff[j, i, length - 1]

        for i in range(maxgrid):
            path[0, i] = mean[0, i]

        for j in range(1, maxgrid):
            for i in range(j, maxgrid):
                path[j, i] = path[j - 1, i - 1] + mean[j, i]

@jit(nopython=True)
def kernel_reg_detect_numba(niter, maxgrid, length, sum_tang, mean, path, diff, sum_diff):
    for t in range(niter):
        for j in range(maxgrid):
            for i in range(j, maxgrid):
                for cnt in range(length):
                    diff[j, i, cnt] = sum_tang[j, i]

        for j in range(maxgrid):
            for i in range(j, maxgrid):
                sum_diff[j, i, 0] = diff[j, i, 0]
                for cnt in range(1, length):
                    sum_diff[j, i, cnt] = sum_diff[j, i, cnt - 1] + diff[j, i, cnt]
                mean[j, i] = sum_diff[j, i, length - 1]

        for i in range(maxgrid):
            path[0, i] = mean[0, i]

        for j in range(1, maxgrid):
            for i in range(j, maxgrid):
                path[j, i] = path[j - 1, i - 1] + mean[j, i]