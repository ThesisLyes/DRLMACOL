import numpy as np
import torch as th
from tqdm import tqdm


size = -1
k = -1
E = -1
N = -1


def tabuEGCP_infeaseable(size_pop, max_iter, A, tColor, vect_fit, alpha, vect_delta, vect_conflicts, tabuTenure, phi, tabuTenureEx):
    Lbound = (np.floor(1.0*size/k))
    Ubound = (np.ceil(1.0*size/k))
    # vect_nb_vois, voisin
    for d in tqdm(range(size_pop)):
        f = 0
        tColor_local = np.zeros((size), np.int16)  # actual coloring
        gamma = np.zeros((size, k), np.int8)
        color_size = np.zeros((k), np.int16)
        nbcfl = 0     # number of conflict / loss function
        for x in range(size):
            for y in range(k):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = -1
            tColor_local[x] = int(tColor[d, x])
            color_size[tColor[d, x]] += 1
        for x in range(size):
            for y in range(x):
                if A[x, y] == 1:
                    gamma[x, tColor_local[y]] += 1
                    gamma[y, tColor_local[x]] += 1
                    if tColor_local[y] == tColor_local[x]:
                        nbcfl += 1
        delta_color_size = ComputeDeltaEquitable(color_size)
        f_best = delta_color_size + phi*nbcfl
        for iter_ in range(max_iter):
            best_delta = 9999
            exchange = False
            best_delta_conflicts = -1
            best_delta_equitable = -1
            best_x = -1
            best_v = -1
            for x in range(size):  # for all vertex
                v_x = tColor_local[x]
                if gamma[x, v_x]:
                    for v in range(k):
                        if v != v_x:
                            delta_equitable = delta_color_size-ComputeDeltaEquitableSimpleChange(
                                color_size, v_x, v)
                            delta_conflicts = gamma[x, v] - gamma[x, v_x]
                            # if v imply less conflict then delta is negatif
                            delta = delta_equitable + phi*delta_conflicts
                            if tabuTenure[d, x, v] <= iter_ or delta + f < f_best:
                                if delta < best_delta:
                                    best_x = x
                                    best_v = v
                                    best_delta = delta
                                    best_delta_conflicts = delta_conflicts
                                    best_delta_equitable = delta_equitable
            for x in range(size):
                if (gamma[x][tColor_local[x]]):
                    v_x = tColor_local[x]
                    for y in range(size):
                        if y != x:
                            delta_conflicts = (gamma[x][tColor_local[y]] - gamma[x][tColor_local[x]]) + (
                                gamma[y][tColor_local[x]] - gamma[y][tColor_local[y]]) - 2*A[x][y]
                            # delta_equitable = ComputeDeltaEquitableExchangeChange(
                            # tColor_local, x, y)
                            delta = phi*delta_conflicts
                            if tabuTenureEx[d, x, y] <= iter_ or delta + f < f_best:
                                if delta < best_delta:
                                    best_x = x
                                    best_y = y
                                    best_delta = delta
                                    best_delta_conflicts = delta_conflicts
                                    best_delta_equitable = 0
                                    exchange = True
            f += best_delta
            nbcfl += best_delta_conflicts
            delta_color_size += best_delta_equitable

            if exchange:
                old_colorx = tColor_local[best_x]
                old_colory = tColor_local[best_y]
                for y in range(size):
                    if A[best_x, y] == 1:
                        gamma[y, old_colorx] -= 1
                        gamma[y, old_colory] += 1
                    if A[best_y, y] == 1:
                        gamma[y, old_colory] -= 1
                        gamma[y, old_colorx] += 1
                tColor_local[best_x] = old_colory
                tColor_local[best_y] = old_colorx
                tabuTenureEx[d, best_x, best_y] = (
                    int(alpha * nbcfl)
                    + int(10 * th.rand(1))
                    + iter_
                )
                if f < f_best:
                    f_best = f
                    for a in range(size):
                        tColor[d, a] = tColor_local[a]
            else:
                old_color = tColor_local[best_x]
                for y in range(size):
                    if A[best_x, y] == 1:
                        gamma[y, old_color] -= 1
                        gamma[y, best_v] += 1
                tColor_local[best_x] = best_v
                tabuTenure[d, best_x, old_color] = (
                    int(alpha * nbcfl)
                    + int(10 * th.rand(1))
                    + iter_
                )
                if f < f_best:
                    f_best = f
                    for a in range(size):
                        tColor[d, a] = tColor_local[a]
        vect_fit[d] = f_best
        vect_conflicts[d] = nbcfl
        vect_delta[d] = delta_color_size


def ComputeDeltaEquitable(color_size, flag):
    color_size_loc = color_size
    delta = 0
    flag = False
    for i in range(k-1):
        for j in range(i+1, k):
            delta_inter = color_size_loc[i]-color_size_loc[j]
            delta += delta_inter
            if delta_inter > 1:
                flag = True
    return delta, flag


def ComputeDeltaEquitableSimpleChange(color_size, old_color, newcolor):
    color_size_loc = color_size
    color_size_loc[old_color] -= 1
    color_size_loc[newcolor] += 1
    delta = 0
    flag = False
    for i in range(k-1):
        for j in range(i+1, k):
            delta_inter = color_size_loc[i]-color_size_loc[j]
            delta += delta_inter
            if delta_inter > 1:
                flag = True
    return delta, flag

# je pense il y a pas besoin puisque un echange change pas la taille des ensembles

# def ComputeDeltaEquitableExchangeChange(tColor, vertex1, vertex2):
#     SizeGroupBefore = np.zeros(k)
#     SizeGroupAfter = np.zeros(k)
#     for i in range(size):
#         SizeGroupBefore[tColor[i]] += 1
#         if i == vertex1:
#             SizeGroupAfter[tColor[vertex2]] += 1
#         elif i == vertex2:
#             SizeGroupAfter[tColor[vertex1]] += 1
#         else:
#             SizeGroupAfter[tColor[i]] += 1
#     delta = np.max(SizeGroupAfter) - np.min(SizeGroupAfter) - \
#         (np.max(SizeGroupBefore) - np.min(SizeGroupBefore))
#     return delta
