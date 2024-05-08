import numpy as np
import torch as th
from tqdm import tqdm


size = -1
k = -1
E = -1
N = -1


def tabuGCP(size_pop, max_iter, A, tColor, vect_fit, alpha, tabuTenure):
    """TODO verify"""
    # vect_nb_vois, voisin
    for d in tqdm(range(size_pop)):
        f = 0
        tColor_local = np.zeros((size), np.int16)
        gamma = np.zeros((size, k), np.int8)
        tColor_cuda = tColor_local
        gamma_cuda = gamma
        for x in range(size):
            for y in range(k):
                gamma_cuda[x, y] = 0
                tabuTenure[d, x, y] = -1
            tColor_cuda[x] = int(tColor[d, x])
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
                            if tabuTenure[d, x, v] <= iter_ or delta + f < f_best:
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
            tabuTenure[d, best_x, old_color] = (
                int(alpha * nbcfl)
                + int(10 * th.rand(1))
                + iter_
            )
            if f < f_best:
                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_cuda[a]
        vect_fit[d] = f_best
