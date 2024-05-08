import numpy as np
import torch as th
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

size = -1
size_pop = -1
k = -1
E = -1
N = -1
max_iter = -1
alpha = -1
A = None
tabuTenure = None


def optimisation(tColor):
    tTenure = np.zeros((size, k), dtype=np.int32)
    f = 0
    tColor_local = np.zeros((size), np.int16)
    gamma = np.zeros((size, k), np.int8)
    tColor_cuda = tColor_local
    gamma_cuda = gamma
    for x in range(size):
        for y in range(k):
            gamma_cuda[x, y] = 0
            tTenure[x, y] = -1
        tColor_cuda[x] = int(tColor[x])
    f = 0
    for x in range(size):
        for y in range(x):
            if A[x, y] == 1:
                gamma_cuda[x, tColor_cuda[y]] += 1
                gamma_cuda[y, tColor_cuda[x]] += 1
                if tColor_cuda[y] == tColor_cuda[x]:
                    f += 1
    f_best = f
    for iter_ in range(max_iter):
        best_delta = 9999
        best_x = -1
        best_v = -1
        nbcfl = 0
        for x in range(size):
            v_x = tColor_cuda[x]
            if gamma_cuda[x, v_x] > 0:
                nbcfl += 1
                for v in range(k):
                    if v != v_x:
                        delta = gamma_cuda[x, v] - gamma_cuda[x, v_x]
                        if tTenure[x, v] <= iter_ or delta + f < f_best:
                            if delta < best_delta:
                                best_x = x
                                best_v = v
                                best_delta = delta
        f += best_delta
        old_color = tColor_cuda[best_x]
        for y in range(size):
            if A[best_x, y] == 1:
                gamma_cuda[y, old_color] -= 1
                gamma_cuda[y, best_v] += 1
        tColor_cuda[best_x] = best_v
        tTenure[best_x, old_color] = (
            int(alpha * nbcfl)
            + int(10 * th.rand(1))
            + iter_
        )
        if f < f_best:
            f_best = f
            for a in range(size):
                tColor[a] = tColor_cuda[a]

    return f_best


def tabuGCP(tColor, vect_fit):
    pool = ThreadPool(6)
    result = pool.map(optimisation, tColor)
    pool.close()
    pool.join()
    for i in range(size_pop):
        vect_fit[i] = result[i]
