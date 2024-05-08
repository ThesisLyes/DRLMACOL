import logging
import math

import torch as th
import numpy as np


size = -1
k = -1
size_pop = -1
E = -1

# Random Initialisation of the popupaltion
def initPopGCP(size_pop, tColor):
    for j in range(size_pop):

        for i in range(size):
            r = int(k * th.rand(1))

            if r >= k:
                r = k - 1

            tColor[j, i] = r

def initPopEGCP(size_pop, tColor):
     equitable=np.zeros(k)
     for j in range(size_pop):

        for i in range(size):
            flag=False
            while not flag:
                r = int(k * th.rand(1))
                if r >= k:
                    r = k - 1
                if (np.min(equitable)==(size//k)) or (equitable[r]<(size//k)):
                    flag=True

            tColor[j, i] = r

# Calculation distance between population and offsprings 
def computeMatrixDistance_PorumbelApprox(
    size_sub_pop, size_sub_pop2, matrixDistance, tColor1, tColor2
):
    for d in range(size_sub_pop * size_sub_pop2):
        idx1 = int(d // size_sub_pop2)
        idx2 = int(d % size_sub_pop2)
        ttNbSameColor = np.zeros((k, k), np.uint8)
        M = np.zeros((k), np.int16)
        sigma = np.zeros((k), np.int16)
        for i in range(k):
            M[i] = 0
            sigma[i] = 0
        for x in range(size):
            ttNbSameColor[int(tColor1[int(idx1), x]),
                          int(tColor2[int(idx2), x])] = 0
        for x in range(size):
            i = int(tColor1[int(idx1), x])
            j = int(tColor2[int(idx2), x])
            ttNbSameColor[i, j] += 1
            if ttNbSameColor[i, j] > M[i]:
                M[i] = ttNbSameColor[i, j]
                sigma[i] = j
        proxi = 0
        for i in range(k):
            proxi += ttNbSameColor[i, sigma[i]]
        matrixDistance[int(idx1), int(idx2)] = size - proxi

# Calculate the distances inside the population 
def computeSymmetricMatrixDistance_PorumbelApprox(size_sub_pop, matrixDistance, tColor):

    for d in range(size_sub_pop * (size_sub_pop - 1) // 2):
        # Get upper triangular matrix indices from thread index !
        idx1 = int(
            size_sub_pop
            - 2
            - int(
                math.sqrt(-8.0 * d + 4.0 * size_sub_pop *
                          (size_sub_pop - 1) - 7) / 2.0
                - 0.5
            )
        )
        idx2 = int(
            d
            + idx1
            + 1
            - size_sub_pop * (size_sub_pop - 1) / 2
            + (size_sub_pop - idx1) * ((size_sub_pop - idx1) - 1) / 2
        )
        ttNbSameColor = np.zeros((k, k), np.uint8)
        M = np.zeros((k), np.int16)
        sigma = np.zeros((k), np.int16)
        for j in range(k):
            M[j] = 0
            sigma[j] = 0
        for x in range(size):
            ttNbSameColor[int(tColor[int(idx1), x]),
                          int(tColor[int(idx2), x])] = 0
        for x in range(size):
            i = int(tColor[int(idx1), x])
            j = int(tColor[int(idx2), x])
            ttNbSameColor[i, j] += 1
            if ttNbSameColor[i, j] > M[i]:
                M[i] = ttNbSameColor[i, j]
                sigma[i] = j
        proxi = 0
        for i in range(k):
            proxi += ttNbSameColor[i, sigma[i]]
        matrixDistance[int(idx1), int(idx2)] = size - proxi
        matrixDistance[int(idx2), int(idx1)] = size - proxi

# Compute the crossovers
def computeClosestCrossover(crossover_number, tColor, allCrossovers, idx1, idx2):
    nbParent = 2
    parents = np.zeros((nbParent, size), np.int16)
    current_child = np.zeros((size), np.int16)
    for j in range(size):
        parents[0, j] = tColor[idx1, j]
        parents[1, j] = tColor[idx2, j]
    for j in range(size):
        current_child[j] = -1
    tSizeOfColors = np.zeros((nbParent, k), np.int16)
    for i in range(nbParent):
        for j in range(k):
            tSizeOfColors[i, j] = 0
        for j in range(size):
            if parents[i, j] > -1:
                tSizeOfColors[i, parents[i, j]] += 1
    for i in range(k):
        indiceParent = i % 2
        valMax = -1
        colorMax = -1
        startColor = int(k * th.rand(1))
        for j in range(k):
            color = (startColor + j) % k
            currentVal = tSizeOfColors[indiceParent, color]
            if currentVal > valMax:
                valMax = currentVal
                colorMax = color
        for j in range(size):
            if parents[int(indiceParent), j] == colorMax and current_child[j] < 0:
                current_child[j] = i
                for l in range(nbParent):
                    if parents[l, j] > -1:
                        tSizeOfColors[l, parents[l, j]] -= 1
    for j in range(size):
        if current_child[j] < 0:
            r = int(k * th.rand(1))
            if r >= k:
                r = k - 1
            current_child[j] = r
    for j in range(size):
        allCrossovers[crossover_number, j] = current_child[j]
    return current_child


def insertion_pop(
    size_pop,
    size_offspring,
    matrixDistanceAll,
    colors_pop,
    offsprings_pop_after_tabu,
    fitness_pop,
    fitness_offsprings_after_tabu,
    matrice_crossovers_already_tested,
    min_dist,
):
    all_scores = np.hstack((fitness_pop, fitness_offsprings_after_tabu))
    matrice_crossovers_already_tested_new = np.zeros(
        (size_pop +size_offspring, size_pop +size_offspring), dtype=np.uint8
    )
    matrice_crossovers_already_tested_new[
        :size_pop, :size_pop
    ] = matrice_crossovers_already_tested
    idx_best = np.argsort(all_scores)
    idx_selected = []
    cpt = 0
    for i in range(0, size_pop +size_offspring):
        idx = idx_best[i]
        if len(idx_selected) > 0:
            dist = np.min(matrixDistanceAll[idx, idx_selected])
        else:
            dist = 9999
        if dist >= min_dist:
            idx_selected.append(idx)
            if idx >= size_pop:
                cpt += 1
        if len(idx_selected) == size_pop:
            break
    logging.info(f"len(idx_selected) {len(idx_selected)}")
    if len(idx_selected) != size_pop:
        for i in range(0, size_pop +size_offspring):
            idx = idx_best[i]
            if idx not in idx_selected:
                dist = np.min(matrixDistanceAll[idx, idx_selected])
                if dist >= 0:
                    idx_selected.append(idx)
            if len(idx_selected) == size_pop:
                break
    logging.info(f"Nb insertion {cpt}")
    new_matrix = matrixDistanceAll[idx_selected, :][:, idx_selected]
    stack_all = np.vstack((colors_pop, offsprings_pop_after_tabu))
    colors_pop_v2 = stack_all[idx_selected]
    fitness_pop_v2 = all_scores[idx_selected]
    matrice_crossovers_already_tested_v2 = matrice_crossovers_already_tested_new[
        idx_selected, :
    ][:, idx_selected]
    return (
        new_matrix,
        fitness_pop_v2,
        colors_pop_v2,
        matrice_crossovers_already_tested_v2,
        cpt,
    )


# (Adjacency Matrix , Array containing the coloring of the population , batch size )
def fit_function(A, tColor, size_batch):
    gamma = np.zeros((size, k), np.int8)
    tColor_local = np.zeros((size), np.int16)
    fit_array = np.ones(size_batch)
    for d in range(size_batch):
        f = 0
        for x in range(size):
            for y in range(k):
                gamma[x, y] = 0
            tColor_local[x] = int(tColor[d][x])
        for x in range(size):
            for y in range(x):
                if A[x, y] == 1:
                    gamma[x, tColor_local[y]] += 1
                    gamma[y, tColor_local[x]] += 1
                    if tColor_local[y] == tColor_local[x]:
                        f += 1

        fit_array[d] = f
    return fit_array

def initGreedyOrderPopWVCP_bigSize(A, W, tColor, fit, gamma):
    for d in range(size_pop):
        for x in range(size):
            for y in range(k):
                gamma[d, x, y] = 0
        f = 0
        nb_max_col = 0
        for x in range(size):
            c = 0
            found = False
            startCol = int(nb_max_col *  th.rand(1))
            while not found and c < nb_max_col:
                v = (startCol + c) % nb_max_col
                if gamma[d, x, v] == 0:
                    found = True
                    tColor[d, x] = v
                    for y in range(size):
                        if A[x, y] == 1:
                            gamma[d, y, v] += 1
                c = c + 1
            if not found:
                if nb_max_col < k:
                    f += W[x]
                    tColor[d, x] = nb_max_col
                    for y in range(size):
                        if A[x, y] == 1:
                            gamma[d, y, nb_max_col] += 1
                    nb_max_col += 1
                else:
                    r = int(k *  th.rand(1))
                    tColor[d, x] = r
        fit[d] = f

def initGreedyOrderPopWVCP( A, W, D, tColor, fit):
    for d in range(size_pop):
        gamma = np.zeros((size, k), np.int8)
        f = 0
        nb_max_col = 0
        for x in range(size):
            c = 0
            found = False
            startCol = int(nb_max_col * th.rand(1))
            while not found and c < nb_max_col:
                v = (startCol + c) % nb_max_col
                if gamma[x, v] == 0:
                    found = True
                    tColor[d, x] = v
                    for y in range(size):
                        if A[x, y] == 1:
                            gamma[y, v] += 1
                c = c + 1
            if not found:
                if nb_max_col < k:
                    f += W[x]
                    tColor[d, x] = nb_max_col
                    for y in range(size):
                        if A[x, y] == 1:
                            gamma[y, nb_max_col] += 1
                    nb_max_col += 1
                else:
                    r = int(k * th.rand(1))
                    tColor[d, x] = r
        fit[d] = f


def insertion_random(size_pop,
    size_offspring,
    A,
    colors_pop,
    fitness_pop,
    matrice_crossovers_already_tested
):  
    logging.info("Start insertion random")
    matrice_crossovers_already_tested_new = np.zeros(
        (size_pop +size_offspring, size_pop +size_offspring), dtype=np.uint8
    )
    matrice_crossovers_already_tested_new[
        :size_pop, :size_pop
    ] = matrice_crossovers_already_tested
    random_budget=size_pop//10 + 1
    fitness_pop_v2=fitness_pop
    colors_pop_v2=colors_pop
    idx =np.argsort(fitness_pop)
    idx_selected=idx[-random_budget:]
    logging.info(f"index selected : {idx_selected}")
    for i in idx_selected:
        for j in range(size):
            r = int(k * th.rand(1))

            if r >= k:
                r = k - 1

            colors_pop_v2[i, j] = r
        fitness_pop_v2[i]=fit_function(A,[colors_pop_v2[i]],1)

        
    matrice_crossovers_already_tested_v2 = matrice_crossovers_already_tested_new[ idx_selected, :][:, idx_selected]
    logging.info(" insertion random end")
    
    return (colors_pop_v2,
            fitness_pop_v2,
            matrice_crossovers_already_tested_v2,
    )
