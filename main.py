import argparse
import datetime
import logging
# import matplotlib.pyplot as plt
import numpy as np
import torch as th

from drlmacol.main_GCP import main_GCP
from drlmacol.main_GCP_parr import main_GCP_parr
from drlmacol.main_random import main_random
from drlmacol.WVCP.main_WVCP import main_WVCP


# import numba as nb

# Parse arguments
parser = argparse.ArgumentParser(description="DMLCOL for GCP ")

# RD is the graph coloring problem without random parents choosing in crossovers
parser.add_argument("problem",  metavar='t', type=str, help="GCP or RD")
parser.add_argument("instance",  metavar='t', type=str, help="instance name")
parser.add_argument("--id_gpu", type=int, help="id_gpu", default=0)
parser.add_argument("--k", type=int, help="number of colors", default=3)
parser.add_argument("--sizepop", type=int,
                    help="size of the population", default=-1)
parser.add_argument(
    "--crossbudget", help="number of crossover to compute", type=int, default=3)

parser.add_argument("--alpha", help="alpha", type=float, default=-1)
parser.add_argument("--nb_neighbors", help="nb_neighbors",
                    type=int, default=-1)
parser.add_argument("--nb_iter_tabu", help="nb_iter_tabu",
                    type=int, default=-1)

parser.add_argument('--test', help="test", action='store_true')
parser.add_argument('--target', help="target of k ", type=int, default=-1)


args = parser.parse_args()

# # Init gpu devices
# nb.cuda.select_device(args.id_gpu)
# device = f"cuda:{args.id_gpu}"
# logging.info(device)

if args.problem == "GCP":
    name_expe = f"GCP_NN__nb_iter_{args.nb_iter_tabu}_k_{args.k}_{args.instance}_{datetime.datetime.now()}.txt"
elif args.problem == "rd":
    name_expe = (
        f"RD__nb_iter_{args.nb_iter_tabu}_k_{args.k}_{args.instance}_{datetime.datetime.now()}.txt"
    )
elif args.problem == "GCPparr":
    name_expe = (
        f"GCPparr__nb_iter_{args.nb_iter_tabu}_k_{args.k}_{args.instance}_{datetime.datetime.now()}.txt"
    )
elif args.problem == "WVCP":
    name_expe = (
        f"WVCP__nb_iter_{args.nb_iter_tabu}_k_{args.k}_{args.instance}_{datetime.datetime.now()}.txt"
    )
else:
    name_expe = (
        f"compare__nb_iter_{args.nb_iter_tabu}_{args.instance}_{datetime.datetime.now()}.txt"
    )

logging.basicConfig(
    handlers=[
        logging.FileHandler(f"logs/{name_expe}.log"),
        logging.StreamHandler(),
    ],
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S",
)


# Init gpu devices

device = "cuda"
logging.info(device)


if args.problem == "GCP":
    for i in range(10):
        logging.info(f"k : {args.k}")
        NN_score1, NN_dist1 = main_GCP(
            args.instance,
            args.k,
            args.alpha if args.alpha != -1 else 0.6,
            args.nb_neighbors if args.nb_neighbors != -1 else 16,
            args.nb_iter_tabu,
            args.test,
            device,
            name_expe,
            args.sizepop,
            args.crossbudget,
        )
        with open("curves" + name_expe, "a", encoding="utf8") as f:
            f.write(
                f"NN_score= {NN_score1} \n NN_dist= {NN_dist1} \n ")
elif args.problem == "GCPparr":
    for i in range(10):
        logging.info(f"k : {args.k}")
        NN_score1, NN_dist1 = main_GCP_parr(
            args.instance,
            args.k,
            args.alpha if args.alpha != -1 else 0.6,
            args.nb_neighbors if args.nb_neighbors != -1 else 16,
            args.nb_iter_tabu,
            args.test,
            device,
            name_expe,
            args.sizepop,
            args.crossbudget,
        )

elif args.problem == "RD":
    logging.info(f"k : {args.k}")
    main_random(
        args.instance,
        args.k,
        args.alpha if args.alpha != -1 else 0.6,
        args.nb_neighbors if args.nb_neighbors != -1 else 16,
        args.nb_iter_tabu,
        args.test,
        device,
        name_expe,
        args.sizepop,
        args.crossbudget,
    )
else:
    if args.problem == "WVCP":
        NN_score1, NN_dist1 = main_WVCP(
            args.instance,
            args.k,
            args.alpha if args.alpha != -1 else 0.2,
            args.nb_neighbors if args.nb_neighbors != -1 else 32,
            args.nb_iter_tabu,
            args.test,
            device,
            name_expe,
            args.sizepop,
            args.crossbudget,
            args.target,
        )
    else:
        for i in range(10):
            th.cuda.empty_cache()
            logging.info(f"k : {args.k}")
            NN_score1, NN_dist1 = main_GCP(
                args.instance,
                args.k,
                args.alpha if args.alpha != -1 else 0.6,
                args.nb_neighbors if args.nb_neighbors != -1 else 16,
                args.nb_iter_tabu,
                args.test,
                device,
                name_expe,
                args.sizepop,
                args.crossbudget,
            )
            th.cuda.empty_cache()

        logging.info(" ")

        logging.info("#################################")
        logging.info("start random")
        logging.info("#################################")

        RD_score1, RD_dist1 = main_random(
            args.instance,
            args.k,
            args.alpha if args.alpha != -1 else 0.6,
            args.nb_neighbors if args.nb_neighbors != -1 else 16,
            args.nb_iter_tabu,
            args.test,
            device,
            name_expe,
            args.sizepop,
            args.crossbudget,
        )

        with open("curves" + name_expe, "a", encoding="utf8") as f:
            f.write(
                f"NN_score= {NN_score1} \n NN_dist= {NN_dist1} \n RD_score={RD_score1} \n RD_dist={RD_dist1} \n")

    # NN_score_mean = np.mean(NN_score,axis=0)
    # NN_dist_mean = np.mean(NN_dist,axis=0)
    # RD_score_mean = np.mean(RD_score,axis=0)
    # RD_dist_mean = np.mean(RD_dist,axis=0)

    # plt.figure(1)
    # plt.plot(range(len(NN_score_mean)), NN_score_mean, label='best score with NN')
    # plt.plot(range(len(RD_score_mean)), RD_score_mean, label='best score without NN')
    # plt.legend(('best score with NN', 'best score without NN'),
    #        loc='upper right', shadow=True)
    # plt.xticks(range(1, max(len(NN_score_mean),len(RD_score_mean)),max(len(NN_score_mean),len(RD_score_mean))//10 +1))
    # plt.xlabel('generation')
    # plt.ylabel("best score")
    # plt.title(f'Impact of NN on score exp_{name_expe}')
    # plt.savefig(f'curves/Curves_GCP_score_{args.instance}_k_{args.k}_{name_expe}.png')

    # plt.figure(2)
    # plt.plot(range(len(NN_dist_mean)), NN_dist_mean, label='best dist with NN')
    # plt.plot(range(len(RD_dist_mean)), RD_dist_mean, label='best dist without NN')
    # plt.legend(('avg dist with NN', 'avg dist without NN'),
    #        loc='upper right', shadow=True)
    # plt.xticks(range(1, max(len(NN_dist_mean),len(RD_dist_mean)),max(len(NN_dist_mean),len(RD_dist_mean))//10 +1))
    # plt.xlabel('generation')
    # plt.ylabel("avg dist")
    # plt.title(f'Impact of NN on distances exp_{name_expe}')
    # plt.savefig(f'curves/Curves_GCP_dist_{args.instance}_k_{args.k}_{name_expe}.png')
