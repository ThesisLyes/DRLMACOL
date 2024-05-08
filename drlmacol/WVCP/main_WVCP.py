import logging
import random
import datetime
from random import shuffle
from time import time
from collections import defaultdict

import numpy as np
import torch as th
from tqdm import tqdm


from drlmacol import memetic_algorithm
from drlmacol.memetic_algorithm import insertion_pop
from drlmacol.NNetWrapper import NNetWrapper as nnetwrapper
from drlmacol.WVCP import tabuCol_WVCP


def test_solution(A, W, S, k):
    nb_conflict = 0
    max_weight = np.zeros((k))
    nb_uncolored = 0
    for v in range(A.shape[0]):
        colorV = S[v]
        if colorV < k:
            if W[v] > max_weight[colorV]:
                max_weight[colorV] = W[v]
            for i in range(A.shape[1]):
                if A[v, i] == 1:
                    colorVprim = S[i]
                    if colorVprim == colorV:
                        nb_conflict += 1
        else:
            nb_uncolored += 1
    score = np.sum(max_weight)
    logging.info(f"nb_conflict : {nb_conflict / 2}")
    logging.info(f"score : {score}")
    logging.info(f"nb_uncolored : {nb_uncolored}")
    return score, nb_conflict / 2


def determine_k(graph):
    size = graph.shape[0]
    vertices = list(range(size))
    max_k = 0
    gamma = np.empty((size, size), dtype=int)
    colors = np.empty(size, dtype=int)
    for _ in range(10):
        random.shuffle(vertices)
        gamma.fill(0)
        colors.fill(-1)
        nb_colors = 0
        for vertex in vertices:
            possible_colors = [
                color for color in range(nb_colors) if gamma[color, vertex] == 0
            ]
            color = random.choice(
                possible_colors) if possible_colors else nb_colors
            if color == nb_colors:
                nb_colors += 1
            colors[vertex] = color
            for v2 in range(size):
                if graph[vertex, v2] == 1:
                    gamma[color, v2] += 1
        max_k = max(max_k, nb_colors)
    return max_k


def main_WVCP(
    instance, k, alpha, nb_neighbors, nb_iter_tabu, test, device, name_expe, size_pop, crossover_budget, target
):

    # get the graph
    filepath = "instances/wvcp_reduced/"
    weights = np.loadtxt(filepath + instance + ".col.w")
    heavyWeights = np.max(weights) >= 128
    size = int(weights.shape[0])
    logging.info(f"size {size}")

    with open(filepath + instance + ".col", "r", encoding="utf8") as f:
        graph = np.zeros((size, size), dtype=np.int16)
        for line in f:
            x = line.split(sep=" ")
            if x[0] == "e":
                graph[int(x[1]) - 1, int(x[2]) - 1] = 1
                graph[int(x[2]) - 1, int(x[1]) - 1] = 1

    beginTime = time()

    A_global_mem = graph
    W_global_mem = weights  # load weights to device

    #########################################

    # Parameters

    max_iter_ITS = 10
    min_dist_insertion = size / 10

    if nb_iter_tabu == -1:
        # nb_iter_tabu = int(size * 128)
        nb_iter_tabu = int(size*4)
        # nb_iter_tabu = size

    if alpha == -1:
        alpha = 0.6

    bigSize = size > 500

    if k == 3:
        k = determine_k(graph)

    if nb_neighbors == -1:
        nb_neighbors = 16

    if test == True:
        logging.info("TEST")

        nb_iter_tabu = 10
        size_pop = 8
        nb_neighbors = 3
        crossover_budget = 8

    best_score = 99999

    # counter of epoch without offspring changes
    count = 0

    # log the size of the coloring we are searching
    logging.info(f"k : {k}")

    #########################################
    #  Init tables

    offsprings_pop = np.zeros(
        (size_pop, size), dtype=np.int32
    )  # new colors generated after offspring

    fitness_pop = (
        np.ones((size_pop), dtype=np.int32) * 9999
    )  # vector of fitness of the population
    fitness_offsprings = np.zeros(
        (size_pop), dtype=np.int32
    )  # vector of fitness of the offsprings

    matrice_crossovers_already_tested = np.zeros(
        (size_pop, size_pop), dtype=np.uint8)

    # Big Distance matrix with all individuals in pop and all offsprings at each generation
    matrixDistanceAll = np.zeros((2 * size_pop, 2 * size_pop), dtype=np.int16)

    matrixDistanceAll[:size_pop, :size_pop] = (
        np.ones((size_pop, size_pop), dtype=np.int16) * 9999
    )

    matrixDistance1 = np.zeros(
        (size_pop, size_pop), dtype=np.int16
    )  # Matrix with distances between individuals in pop and offsprings
    matrixDistance2 = np.zeros(
        (size_pop, size_pop), dtype=np.int16
    )  # Matrix with distances between all offsprings

    vect_score = np.zeros((size_pop), dtype=np.float32)
    vect_conflicts = np.zeros((size_pop), dtype=np.float32)

    tabuTenure = np.zeros((size_pop, size, k), dtype=np.int32)

    # Array with the results of crossovers opperations
    crossovers_np = np.zeros((crossover_budget, size), dtype=np.int32)

    memetic_algorithm.size = size
    memetic_algorithm.k = k
    tabuCol_WVCP.size = size
    tabuCol_WVCP.k = k
    memetic_algorithm.size_pop = size_pop
    # if "inithx" in instance or "zeroin" in instance:
    #     tabuCol_WVCP.N = int(size // k * 20)
    # else:
    #     tabuCol_WVCP.N = int(size // k * 12)
    tabuCol_WVCP.N = int(size)

    # List for tests

    best_scores = []
    avg_dists = []
    saved = []
    #########################################

    # Init population

    logging.info("initPopGCP")
    if bigSize:
        gamma_tabucol = np.zeros((size_pop, size, k), dtype=np.int8)
        memetic_algorithm.initGreedyOrderPopWVCP_bigSize(
            A_global_mem,
            W_global_mem,
            size_pop,
            offsprings_pop,
            fitness_offsprings,
            gamma_tabucol,
        )
    else:
        memetic_algorithm.initGreedyOrderPopWVCP(
            A_global_mem,
            W_global_mem,
            size_pop,
            offsprings_pop,
            fitness_offsprings,
        )

    phi = int(k / size * np.max(weights)) / 2.0
    if phi == 0:
        phi = 0.5

    vect_phi = np.ones((size_pop), dtype=np.float32) * phi

    colors_pop = offsprings_pop
    score, nb_conflict = test_solution(graph, weights, colors_pop[0], k)

    logging.info(f"score test : {score}")
    logging.info(f"nb_conflict test : {nb_conflict}")

    th.cuda.empty_cache()

    #########################################
    #  Create the neural networks
    crossover_policy_net_1 = nnetwrapper(
        size,
        k,
        dropout=0.0,
        remix=True,
        verbose=False,
        layers_size=[
            size,
            size * 10,
            size * 5,
            size,
            size // 2,
            1,
        ],)
    crossover_policy_net_1.set_to_device(device)

    crossover_optimizer = th.optim.Adam(list(crossover_policy_net_1.parameters(
    )), lr=1e-3)

    #########################################
    # First step : local search
    # Start tabu

    logging.info("############################")
    logging.info("Start TABU")
    logging.info("############################")

    start = time()
    # Init the result matrix
    offsprings_pop_after_tabu = np.zeros((size_pop, size), dtype=np.int32)
    fitness_offsprings_after_tabu = np.ones(
        (size_pop), dtype=np.int32) * 9999999999
    # Collect the starting points of the local search and convert it into torch tensor - X's of the training dataset

    for i in range(max_iter_ITS):
        logging.info(f"iter tabu : {i}")
        logging.info("stats phi")
        logging.info(f"min {np.min(vect_phi)}")
        logging.info(f"mean {np.mean(vect_phi)}")
        logging.info(f"max {np.max(vect_phi)}")
        if i == max_iter_ITS - 1:
            vect_phi_global_mem = (
                np.ones((size_pop), dtype=np.float32) * np.max(weights) * 2
            )
        else:
            vect_phi_global_mem = (vect_phi)

        if bigSize:
            tabuCol_WVCP.tabuWVCP_NoRandom_AFISA_bigSize
            (
                size_pop,
                nb_iter_tabu,
                A_global_mem,
                W_global_mem,
                offsprings_pop,
                fitness_offsprings,
                vect_score,
                vect_conflicts,
                alpha,
                vect_phi_global_mem,
                gamma_tabucol,
            )
        else:
            if heavyWeights:
                tabuCol_WVCP.tabuWVCP_NoRandom_AFISA_heavyWeights(
                    size_pop,
                    nb_iter_tabu,
                    A_global_mem,
                    W_global_mem,
                    offsprings_pop,
                    fitness_offsprings,
                    vect_score,
                    vect_conflicts,
                    alpha,
                    vect_phi_global_mem,
                )
            else:
                tabuCol_WVCP.tabuWVCP_NoRandom_AFISA(
                    size_pop,
                    nb_iter_tabu,
                    A_global_mem,
                    W_global_mem,
                    offsprings_pop,
                    fitness_offsprings,
                    vect_score,
                    vect_conflicts,
                    alpha,
                    vect_phi_global_mem,
                )

        offsprings_pop_after_tabu = offsprings_pop

        logging.info(f"vect_score {vect_score}")
        logging.info(f"vect_conflicts {vect_conflicts}")
        logging.info(f"Tabucol duration : {time() - start}")
        for d in range(size_pop):
            if vect_conflicts[d] == 0:
                vect_phi[d] = vect_phi[d] / 2
                if fitness_offsprings[d] < fitness_offsprings_after_tabu[d]:
                    fitness_offsprings_after_tabu[d] = fitness_offsprings[d]
                    offsprings_pop_after_tabu[d, :] = offsprings_pop[d, :]
            else:
                vect_phi[d] = vect_phi[d] * 2

        best_score_pop = np.min(fitness_offsprings_after_tabu)
        avg_pop = np.mean(fitness_offsprings_after_tabu)

        b = np.argwhere(vect_score <= target)
        c = vect_conflicts[b]
        d = np.argwhere(c == 0)[:, 0]
        e = b[d]
        # print([b, c, d, e])
        # print(best_score_pop)
        if best_score_pop <= target and e.size > 0:
            min_index = np.argmin(vect_score[e])
            solution_score = vect_score[e[min_index]]
            if not (solution_score in saved):
                logging.info("Save best solution 1")
                saved.append(vect_score[e[min_index]])
                solution = offsprings_pop_after_tabu[
                    e[min_index]
                ]
                with open("evol/WVCP/WNN/" + f'k ={k}'+name_expe, "a", encoding="utf8") as f:
                    f.write(
                        f"{solution_score},{avg_pop},{-1},{time() - beginTime}\n")

                np.savetxt(
                    f"solutions/Solutions_WVCP_{instance}_k_{k}_score_{solution_score}_epoch_{-1}_{datetime.datetime.now()}_{time()-beginTime}.csv",
                    solution.astype(int),
                    fmt="%i",
                )

            # return best_score_pop, [0]

    best_score_pop = np.min(fitness_offsprings_after_tabu)
    worst_score_pop = np.max(fitness_offsprings_after_tabu)
    avg_pop = np.mean(fitness_offsprings_after_tabu)

    logging.info(
        f"Pop best : {best_score_pop}_worst : {worst_score_pop}_avg : {avg_pop}"
    )

    logging.info("end tabu")

    # Get and log results

    logging.info("############################")
    logging.info("Results TabuCol")
    logging.info("############################")

    best_current_score = min(fitness_offsprings_after_tabu)

    if best_current_score < best_score:

        best_score = best_current_score

        # logging.info("Save best solution")

        # solution = offsprings_pop_after_tabu[
        #     np.argmin(fitness_offsprings_after_tabu)
        # ]

        # np.savetxt(
        #     f"solutions/Solutions_GCP_{instance}_k_{k}_score_{best_current_score}_epoch_{-1}.csv",
        #     solution.astype(int),
        #     fmt="%i",
        # )

    with open("evol/WVCP/WNN/" + f'k ={k}'+name_expe, "a", encoding="utf8") as f:
        f.write(
            f"{best_score},{best_current_score},{-1},{time() - beginTime}\n")

    # if best_score <= target and not vect_conflicts[np.argmin(fitness_pop)]:
    #     logging.info("Save best solution 2")

    #     solution = offsprings_pop_after_tabu[
    #         np.argmin(fitness_offsprings_after_tabu)
    #     ]

    #     np.savetxt(
    #         f"solutions/Solutions_WVCP_{instance}_k_{k}_score_{best_current_score}_epoch_{-1}.csv",
    #         solution.astype(int),
    #         fmt="%i",
    #     )

        # return best_score, [0]

    # Second step : insertion of offsprings in pop according to diversity/fit criterion

    logging.info("Keep best with diversity/fit tradeoff")

    ########################################
    logging.info("start matrix distance")

    start = time()

    offsprings_pop = offsprings_pop_after_tabu

    memetic_algorithm.computeMatrixDistance_PorumbelApprox(
        size_pop,
        size_pop,
        matrixDistance1,
        colors_pop,
        offsprings_pop,
    )
    matrixDistance1 = matrixDistance1

    memetic_algorithm.computeSymmetricMatrixDistance_PorumbelApprox(
        size_pop, matrixDistance2, offsprings_pop
    )

    matrixDistance2 = matrixDistance2
    matrixDistanceAll[:size_pop, size_pop:] = matrixDistance1
    matrixDistanceAll[size_pop:,
                      :size_pop] = matrixDistance1.transpose(1, 0)
    matrixDistanceAll[size_pop:, size_pop:] = matrixDistance2

    offsprings_pop = None

    logging.info("end  matrix distance")
    #####################################

    logging.info("start insertion in pop")
    start = time()

    results = insertion_pop(
        size_pop,
        size_pop,
        matrixDistanceAll,
        colors_pop,
        offsprings_pop_after_tabu,
        fitness_pop,
        fitness_offsprings_after_tabu,
        matrice_crossovers_already_tested,
        min_dist_insertion,
    )

    matrixDistanceAll[:size_pop, :size_pop] = results[0]
    fitness_pop = results[1]
    colors_pop = results[2]
    matrice_crossovers_already_tested = results[3]
    if results[4] == 0:
        count += 1

    logging.info(f"Insertion in pop : {time() - start}")

    logging.info("end insertion in pop")

    logging.info("After keep best info")

    best_score_pop = np.min(fitness_pop)
    worst_score_pop = np.max(fitness_pop)
    avg_score_pop = np.mean(fitness_pop)

    logging.info(
        f"Pop _best : {best_score_pop}_worst : {worst_score_pop}_avg : {avg_score_pop}"
    )
    logging.info(fitness_pop)
    matrix_distance_pop = matrixDistanceAll[:size_pop, :size_pop]
    max_dist = np.max(matrix_distance_pop)
    min_dist = np.min(matrix_distance_pop + np.eye(size_pop) * 9999)
    avg_dist = np.sum(matrix_distance_pop) / (size_pop * (size_pop - 1))
    logging.info(
        f"Avg dist : {avg_dist} min dist : {min_dist} max dist : {max_dist}"
    )

    for epoch in range(100000):
        performance = 0
        th.cuda.empty_cache()

        # Third step : selection of best crossovers to generate new offsprings
        logging.info("############################")
        logging.info("start crossover")
        logging.info("############################")

        bestColor_global_mem = colors_pop

        graphs = th.FloatTensor(colors_pop.astype(np.float32))
        zeros = th.zeros((graphs.size()[0], size, k))
        ones = th.ones((graphs.size()[0], size, k))
        graphs = zeros.scatter_(2, graphs.unsqueeze(2).long(), ones)
        crossover_number = 0
        with th.no_grad():
            if epoch > 5:
                # evaluate probability distribution in RGA
                _, crossover_sample_probability_list = crossover_policy_net_1.forward_batch(
                    graphs)

            else:
                crossover_sample_probability_list = [
                    1.0/size_pop for _ in range(size_pop)]

            sampled_idx = random.choices(list(range(len(crossover_sample_probability_list))),
                                         weights=crossover_sample_probability_list,
                                         k=crossover_budget)

            crossover_train_data = np.ones(
                (crossover_budget, 4), dtype=int)*(-1)

            # outer loop, first parent

            for idx in tqdm(sampled_idx):

                holdout_colors_list = np.zeros(
                    (size_pop-1, size), dtype=np.int32
                )
                flag = False
                for i in range(size_pop-1):
                    if flag:
                        holdout_colors_list[i] = colors_pop[i+1]
                    elif i == idx:
                        flag = True
                        holdout_colors_list[i] = colors_pop[i+1]
                    else:
                        holdout_colors_list[i] = colors_pop[i]

                if holdout_colors_list.size == 0:
                    logging.info("Holdoutlist empty")
                    continue
                holdout = th.FloatTensor(
                    holdout_colors_list.astype(np.float32))
                zeros = th.zeros((holdout.size()[0], size, k))
                ones = th.ones((holdout.size()[0], size, k))
                holdout = zeros.scatter_(2, holdout.unsqueeze(2).long(), ones)
                if epoch > 5:
                    # evaluate probability distribution for second parent
                    _, crossover_sample_probability_list_2 = crossover_policy_net_1.forward_batch(
                        holdout)
                else:
                    crossover_sample_probability_list_2 = [
                        1.0/len(holdout_colors_list) for i in range(len(holdout_colors_list))]

                # uncomment to test if the networks have an impact (have to comment the if else close upthere)
                # crossover_sample_probability_list_2 = [1.0/len(holdout_colors_list) for i in range(len(holdout_colors_list))]

                sampled_idx_2 = random.choices(list(range(len(holdout_colors_list))),
                                               weights=crossover_sample_probability_list_2, k=size_pop)
                # inner loop, second parent
                for idx2 in sampled_idx_2:
                    if not matrice_crossovers_already_tested[idx, idx2]:
                        matrice_crossovers_already_tested[idx, idx2] = 1
                    else:
                        continue
                    real_idx2 = 0
                    if idx2 < idx:
                        real_idx2 = idx2
                    else:
                        real_idx2 = idx2+1

                    memetic_algorithm.computeClosestCrossover(
                        crossover_number,
                        bestColor_global_mem,
                        crossovers_np,
                        idx,
                        real_idx2,
                    )

                    crossover_train_data[crossover_number] = [idx, idx2,
                                                              fitness_pop[idx], fitness_pop[real_idx2]]  # log_crossover_sample_probability[idx]/fitting scores
                    crossover_number += 1
                    break

                ################################################
                ################################################
                ################################################
                ################################################
                ################################################
                ################################################

            logging.info(
                f"nb cross already tested in pop : {np.sum(matrice_crossovers_already_tested)}"
            )
            logging.info("############################")
            logging.info("Start TABU crossovers")
            logging.info("############################")

            vect_phi_cross = np.ones((size_pop), dtype=np.float32) * phi

            start = time()
            # Init the result matrix
            crossovers_pop_after_tabu = np.zeros(
                (crossover_budget, size), dtype=np.int32)
            fitness_crossovers_after_tabu = np.ones(
                (crossover_budget), dtype=np.int32) * 99999
            fitness_crossovers = np.ones(
                (crossover_budget), dtype=np.int32) * 99999
            for j in range(max_iter_ITS):
                logging.info(f"iter tabu : {j}")
                logging.info("stats phi")
                logging.info(f"min {np.min(vect_phi_cross)}")
                logging.info(f"mean {np.mean(vect_phi_cross)}")
                logging.info(f"max {np.max(vect_phi_cross)}")

                vect_phi_cross_global_mem = (vect_phi_cross)

                if bigSize:
                    tabuCol_WVCP.tabuWVCP_NoRandom_AFISA_bigSize
                    (
                        size_pop,
                        nb_iter_tabu,
                        A_global_mem,
                        W_global_mem,
                        crossovers_np,
                        fitness_crossovers_after_tabu,
                        vect_score,
                        vect_conflicts,
                        alpha,
                        vect_phi_cross_global_mem,
                        gamma_tabucol,
                    )
                else:
                    if heavyWeights:
                        tabuCol_WVCP.tabuWVCP_NoRandom_AFISA_heavyWeights(
                            size_pop,
                            nb_iter_tabu,
                            A_global_mem,
                            W_global_mem,
                            crossovers_np,
                            fitness_crossovers_after_tabu,
                            vect_score,
                            vect_conflicts,
                            alpha,
                            vect_phi_cross_global_mem,
                        )
                    else:
                        tabuCol_WVCP.tabuWVCP_NoRandom_AFISA(
                            size_pop,
                            nb_iter_tabu,
                            A_global_mem,
                            W_global_mem,
                            crossovers_np,
                            fitness_crossovers_after_tabu,
                            vect_score,
                            vect_conflicts,
                            alpha,
                            vect_phi_cross_global_mem,
                        )
                    fitness_crossovers_after_tabu = fitness_crossovers_after_tabu
                    size_crossovers = len(crossovers_np)
                    crossovers_pop_after_tabu = crossovers_np
                    logging.info(f"vect_score {vect_score}")
                    logging.info(f"vect_conflicts {vect_conflicts}")
                    logging.info(f"Tabucol duration : {time() - start}")
                    if j == 0:
                        fitness_crossovers = fitness_crossovers_after_tabu
                    else:
                        for d in range(size_pop):
                            if vect_conflicts[d] == 0:
                                vect_phi_cross[d] = vect_phi_cross[d] / 2
                                if fitness_crossovers[d] < fitness_crossovers_after_tabu[d]:
                                    fitness_crossovers_after_tabu[d] = fitness_crossovers[d]
                                    crossovers_pop_after_tabu[d,
                                                              :] = crossovers_np[d, :]
                            else:
                                vect_phi_cross[d] = vect_phi_cross[d] * 2
                best_score_pop = np.min(fitness_crossovers_after_tabu)

                b = np.argwhere(vect_score <= target)
                c = vect_conflicts[b]
                d = np.argwhere(c == 0)[:, 0]
                e = b[d]
                # print([b, c, d, e])
                # print(best_score_pop)
                if best_score_pop <= target and e.size > 0:
                    min_index = np.argmin(vect_score[e])
                    solution_score = vect_score[e[min_index]]
                    if not (solution_score in saved):
                        logging.info("Save best solution 3")
                        saved.append(vect_score[e[min_index]])

                        logging.info(f"{solution_score}")
                        solution = crossovers_pop_after_tabu[
                            e[min_index]
                        ]

                        np.savetxt(
                            f"solutions/Solutions_WVCP_{instance}_k_{k}_score_{solution_score}_epoch_{epoch}_{datetime.datetime.now()}_{time()-beginTime}.csv",
                            solution.astype(int),
                            fmt="%i",
                        )

                    # return best_score_pop, [0]

            size_crossovers = len(crossovers_np)

            # Insertion of offsprings in pop according to diversity/fit criterion
            matrixDistance3 = np.zeros(
                (size_pop, size_crossovers), dtype=np.int16
            )  # Matrix with distances between individuals in pop and offsprings
            matrixDistance4 = np.zeros(
                (size_crossovers, size_crossovers), dtype=np.int16
            )  # Matrix with distances between all offsprings
            logging.info("Keep best with diversity/fit tradeoff")

            matrixDistanceAll2 = np.zeros(
                (size_pop+size_crossovers,  size_pop+size_crossovers), dtype=np.int16)

            ########################################
            logging.info("start matrix distance")

            start = time()

            memetic_algorithm.computeMatrixDistance_PorumbelApprox(
                size_pop,
                size_crossovers,
                matrixDistance3,
                colors_pop,
                crossovers_pop_after_tabu,
            )

            memetic_algorithm.computeSymmetricMatrixDistance_PorumbelApprox(
                size_crossovers, matrixDistance4, crossovers_pop_after_tabu
            )

            matrixDistanceAll2[:size_pop,
                               :size_pop] = matrixDistanceAll[:size_pop, :size_pop]
            matrixDistanceAll2[:size_pop, size_pop:] = matrixDistance3
            matrixDistanceAll2[size_pop:,
                               :size_pop] = matrixDistance3.transpose(1, 0)
            matrixDistanceAll2[size_pop:, size_pop:] = matrixDistance4

            logging.info("end  matrix distance")

            ################################################
            ################################################
            ################################################
            ################################################
            ################################################
            ################################################
        if epoch > 3:
            th.cuda.empty_cache()
            logging.info("############################")
            logging.info("crossover policy network optimization")
            logging.info("############################")
            c = 0
            logging.info('crossover_train_data')
            logging.info(crossover_train_data)
            for (idx, idx2, fittness_par1, fittness_par2) in crossover_train_data:
                crossover_optimizer.zero_grad()

                # calculationg log likelihood
                log_likelihood_1, _ = crossover_policy_net_1.forward_batch(
                    graphs)

                avg_dist_1 = np.average(
                    matrixDistanceAll[idx, [i for i in range(size_pop)]])
                avg_dist_2 = np.average(
                    matrixDistanceAll[idx2+1, [i for i in range(size_pop)]])
                avg_dist_child = np.average(
                    matrixDistanceAll2[size_pop+c, [i for i in range(size_pop)]])

                holdout_colors_list = np.zeros(
                    (size_pop-1, size), dtype=np.int32
                )

                flag = False
                for i in range(size_pop-1):
                    if flag:
                        holdout_colors_list[i] = colors_pop[i+1]
                    elif i == idx:
                        flag = True
                        holdout_colors_list[i] = colors_pop[i+1]
                    else:
                        holdout_colors_list[i] = colors_pop[i]

                if holdout_colors_list.size == 0:
                    logging.info("Holdoutlist empty")
                holdout = th.FloatTensor(
                    holdout_colors_list.astype(np.float32))
                zeros = th.zeros((holdout.size()[0], size, k))
                ones = th.ones((holdout.size()[0], size, k))
                holdout = zeros.scatter_(2, holdout.unsqueeze(2).long(), ones)
                log_likelihood_2, _ = crossover_policy_net_1.forward_batch(
                    holdout)
                # reward= 100*((-fitness_crossovers_after_tabu[c] - max (-fittness_par1, -fittness_par2))/(worst_score_pop - best_score_pop))
                # reward = ( fitness_crossovers_after_tabu[c] - min(-fittness_par1, -fittness_par2)) +(-avg_dist_child - max(avg_dist_1,avg_dist_2))
                # reward = - \
                #     fitness_crossovers_after_tabu[c] - \
                #     max(-fittness_par1, -fittness_par2)
                reward = 100*(((-fitness_crossovers_after_tabu[c] - max(-fittness_par1, -fittness_par2))/(worst_score_pop - best_score_pop))
                              + ((avg_dist_child - min(avg_dist_1, avg_dist_2))/(max_dist - min_dist)))/2
                logging.info(f"reward: {reward}")
                # policy gradient
                log_likelihood = -(
                    log_likelihood_1[idx] + log_likelihood_2[idx2]) * reward
                # back propagation
                log_likelihood.backward()
                crossover_optimizer.step()
                performance += reward
                c += 1
                th.cuda.empty_cache()

            logging.info(f"c = {c}")
        logging.info("############################")
        logging.info("Choosing offspring")
        logging.info("############################")

        logging.info("start insertion in pop")
        start = time()
        results = insertion_pop(
            size_pop,
            crossover_budget,
            matrixDistanceAll2,
            colors_pop,
            crossovers_pop_after_tabu,
            fitness_pop,
            fitness_crossovers_after_tabu,
            matrice_crossovers_already_tested,
            min_dist_insertion,
        )

        matrixDistanceAll[:size_pop, :size_pop] = results[0]
        fitness_pop = results[1]
        colors_pop = results[2]
        offsprings_pop = colors_pop
        matrice_crossovers_already_tested = results[3]
        if results[4] == 0:
            count += 1
        else:
            count = 0
        # if crossover_number <crossover_budget//3:
        #              matrice_crossovers_already_tested = np.zeros(
        #                     (size_pop, size_pop), dtype=np.uint8)

        logging.info(f"Insertion in pop : {time() - start}")

        logging.info("end insertion in pop")

        logging.info("After keep best info")

        best_score_pop = np.min(fitness_pop)
        worst_score_pop = np.max(fitness_pop)
        avg_score_pop = np.mean(fitness_pop)

        logging.info(
            f"Pop _best : {best_score_pop}_worst : {worst_score_pop}_avg : {avg_score_pop}"
        )
        logging.info(fitness_pop)
        matrix_distance_pop = matrixDistanceAll[:size_pop, :size_pop]
        max_dist = np.max(matrix_distance_pop)
        min_dist = np.min(matrix_distance_pop + np.eye(size_pop) * 9999)
        avg_dist = np.sum(matrix_distance_pop) / (size_pop * (size_pop - 1))
        logging.info(
            f"Avg dist : {avg_dist} min dist : {min_dist} max dist : {max_dist}"
        )
        logging.info("############################")
        logging.info("Start TABU")
        logging.info("############################")

        start = time()
        # Init the result matrix
        offsprings_pop_after_tabu = np.zeros((size_pop, size), dtype=np.int32)
        fitness_offsprings_after_tabu = np.ones(
            (size_pop), dtype=np.int32) * 9999999999
        # Collect the starting points of the local search and convert it into torch tensor - X's of the training dataset

        for i in range(max_iter_ITS):
            logging.info(f"iter tabu : {i}")
            logging.info("stats phi")
            logging.info(f"min {np.min(vect_phi)}")
            logging.info(f"mean {np.mean(vect_phi)}")
            logging.info(f"max {np.max(vect_phi)}")
            if i == max_iter_ITS - 1:
                vect_phi_global_mem = (
                    np.ones((size_pop), dtype=np.float32) * np.max(weights) * 2
                )
            else:
                vect_phi_global_mem = (vect_phi)

            if bigSize:
                tabuCol_WVCP.tabuWVCP_NoRandom_AFISA_bigSize
                (
                    size_pop,
                    nb_iter_tabu,
                    A_global_mem,
                    W_global_mem,
                    offsprings_pop,
                    fitness_offsprings,
                    vect_score,
                    vect_conflicts,
                    alpha,
                    vect_phi_global_mem,
                    gamma_tabucol,
                )
            else:
                if heavyWeights:
                    tabuCol_WVCP.tabuWVCP_NoRandom_AFISA_heavyWeights(
                        size_pop,
                        nb_iter_tabu,
                        A_global_mem,
                        W_global_mem,
                        offsprings_pop,
                        fitness_offsprings,
                        vect_score,
                        vect_conflicts,
                        alpha,
                        vect_phi_global_mem,
                    )
                else:
                    tabuCol_WVCP.tabuWVCP_NoRandom_AFISA(
                        size_pop,
                        nb_iter_tabu,
                        A_global_mem,
                        W_global_mem,
                        offsprings_pop,
                        fitness_offsprings,
                        vect_score,
                        vect_conflicts,
                        alpha,
                        vect_phi_global_mem,
                    )

            offsprings_pop_after_tabu = offsprings_pop

            logging.info(f"vect_score {vect_score}")
            logging.info(f"vect_conflicts {vect_conflicts}")
            logging.info(f"Tabucol duration : {time() - start}")
            for d in range(size_pop):
                if vect_conflicts[d] == 0:
                    vect_phi[d] = vect_phi[d] / 2
                    if fitness_offsprings[d] < fitness_offsprings_after_tabu[d]:
                        fitness_offsprings_after_tabu[d] = fitness_offsprings[d]
                        offsprings_pop_after_tabu[d, :] = offsprings_pop[d, :]
                else:
                    vect_phi[d] = vect_phi[d] * 2

            best_score_pop = np.min(fitness_offsprings_after_tabu)
            avg_pop = np.mean(fitness_offsprings_after_tabu)

            b = np.argwhere(vect_score <= target)
            c = vect_conflicts[b]
            d = np.argwhere(c == 0)[:, 0]
            e = b[d]
            # print([b, c, d, e])
            # print(best_score_pop)
            if best_score_pop <= target and e.size > 0:
                min_index = np.argmin(vect_score[e])
                if not (vect_score[e[min_index]] in saved):
                    logging.info("Save best solution 4")
                    saved.append(vect_score[e[min_index]])
                    solution_score = vect_score[e[min_index]]
                    logging.info(f"{solution_score}")
                    solution = offsprings_pop_after_tabu[
                        e[min_index]
                    ]
                    with open("evol/WVCP/WNN/" + f'k ={k}'+name_expe, "a", encoding="utf8") as f:
                        f.write(
                            f"{solution_score},{avg_pop},{epoch},{time() - beginTime}\n")

                    np.savetxt(
                        f"solutions/Solutions_WVCP_{instance}_k_{k}_score_{solution_score}_epoch_{epoch}_{datetime.datetime.now()}_{time()-beginTime}.csv",
                        solution.astype(int),
                        fmt="%i",
                    )

                    # return best_score_pop, [0]

        if epoch > 3:
            avg_dists.append(avg_dist)

        ##############################################
        best_current_score = min(fitness_pop)

        if best_current_score < best_score:

            best_score = best_current_score

            # logging.info("Save best solution")

            # solution = offsprings_pop[
            #     np.argmin(fitness_pop)
            # ]

            # np.savetxt(
            #     f"solutions/Solutions_GCP_{instance}_k_{k}_score_{best_current_score}_epoch_{epoch}_after_crossovers.csv",
            #     solution.astype(int),
            #     fmt="%i",
            # )

        with open("evol/WVCP/WNN/" + f'k ={k}'+name_expe, "a", encoding="utf8") as f:
            f.write(
                f"{best_score},{best_current_score},{epoch},{time() - beginTime}\n")
        if epoch > 3:
            best_scores.append(best_score)

        # if best_score <= target and not vect_conflicts[np.argmin(fitness_pop)]:

        #     logging.info("Save best solution (last)")

        #     solution = offsprings_pop[
        #         np.argmin(fitness_pop)
        #     ]

        #     np.savetxt(
        #         f"solutions/Solutions_GCP_{instance}_k_{k}_score_{best_current_score}_epoch_{epoch}_after_crossovers.csv",
        #         solution.astype(int),
        #         fmt="%i",
        #     )
        #     return best_scores, avg_dists

        logging.info(f"count: {count}")
        with open("curves" + name_expe, "a", encoding="utf8") as f:
            f.write(
                f"perf_{epoch}= {performance} \n ")
        performance = 0

    return best_scores, avg_dists
    # th.save(crossover_policy_net_1, 'save_model/crossover_policy_net_1.ckpt')
