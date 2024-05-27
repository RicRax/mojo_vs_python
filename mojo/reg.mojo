
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

def benchmark_reg(niter, maxgrid, length):
    sum_tang = matrix_init(ni,nk)
    mean = matrix_init(nk,nj)

    for i in range(ni):
        tmp_row = object([])
        for j in range(nj):
            tmp_row.append(0.0)
        tmp.append(tmp_row)

    alpha = 32412
    beta = 2123


    var prev = now()
    kernel_2mm(A,B,C,D,ni,nj,nk,nl,alpha,beta,tmp)
    var curr = now()
    var res = curr - prev
    _ = (A, B, C, D)

    return res
    