import csv
import numpy as np


def read_coloring_from_csv(file_path):
    coloring = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for col in reader:
            coloring.extend(map(int, col))
    return coloring


def is_legal_coloring(adjacency_matrix, coloring):
    num_vertices = len(adjacency_matrix)

    # Iterate through each vertex
    for vertex in range(num_vertices):
        # Check if any adjacent vertices have the same color
        for neighbor in range(num_vertices):
            if adjacency_matrix[vertex][neighbor] == 1 and coloring[vertex] == coloring[neighbor]:
                return False
    return True


def test_solution(adjacency_matrix, weights, coloring):
    nb_conflict = 0

    max_weight = np.zeros((np.max(coloring)+1))
    nb_uncolored = 0
    for v in range(adjacency_matrix.shape[0]):
        colorV = coloring[v]

        if weights[v] > max_weight[colorV]:
            max_weight[colorV] = weights[v]
        for i in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[v, i] == 1:
                colorVprim = coloring[i]
                if colorVprim == colorV:
                    nb_conflict += 1

    score = np.sum(max_weight)
    print(f"nb_conflict : {nb_conflict / 2}")
    print(f"score : {score}")
    print(f"nb_uncolored : {nb_uncolored}")
    return score, nb_conflict / 2


coloring_file_path = 'solutions/Solutions_WVCP_DSJC125.1gb_r_k_9_score_90_epoch_-1.csv'
filepath = "instances/wvcp_reduced/"
instance = "DSJC125.1gb_r"

weights = np.loadtxt(filepath + instance + ".col.w")
print(weights)
with open(filepath + instance + ".col", "r", encoding="utf8") as f:
    for line in f:
        x = line.split(sep=" ")
        if x[0] == "p":
            size = int(x[2])
            break

    graph = np.zeros((size, size), dtype=np.int16)

    for line in f:
        x = line.split(sep=" ")
        if x[0] == "e":
            graph[int(x[1]) - 1, int(x[2]) - 1] = 1
            graph[int(x[2]) - 1, int(x[1]) - 1] = 1
adjacency_matrix = graph
coloring = read_coloring_from_csv(coloring_file_path)
print(np.max(coloring))
test_solution(adjacency_matrix, weights, coloring)
if is_legal_coloring(adjacency_matrix, coloring):
    print("The coloring is legal.")
else:
    print("The coloring is not legal.")
