import csv
import numpy as np


def read_coloring_from_csv(file_path):
    coloring = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            coloring.extend(map(int, row))
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


coloring_file_path = 'solutions/Solutions_GCP_DSJR500.1_k_12_score_0_epoch_0_after_crossovers.csv'
filepath = "instances/gcp/"
instance = "DSJR500.1"


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


if is_legal_coloring(adjacency_matrix, coloring):
    print("The coloring is legal.")
else:
    print("The coloring is not legal.")
