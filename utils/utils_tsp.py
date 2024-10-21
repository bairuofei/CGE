import random
import time
import numpy as np
import networkx as nx

from concorde import Problem, run_concorde  

from utils.utils_graph import generate_random_gridmap, get_distance_matrix_for_new_tsp_solver,\
    random_remove_edges, random_remove_nodes, add_position_attr_to_graph, add_weight_attr_to_graph,\
    add_edge_information_matrix, add_graph_weights_as_dopt, connect_tsp_path
from utils.utils_visualization import plot_modified_path

def symmetricize(matrix, k=None):
    """
        Jonker-Volgenant method of transforming (n x n) asymmetric TSP, C into (2n x 2n) symmetric TSP, S.

        Let C be an asymmetric TSP matrix.
        Let k be a very large number, ie k >> C.max()
        Let U = (u_ij) with u_ii = 0, u_ij = k for i != j.

        Construct the (2n x 2n) matrix:
        
                    +-------+
                    | U |C^T|
            S =     |---+---|
                    | C | U |
                    +-------+

        S is a symmetric TSP problem on 2n nodes.
        There is a one-to-one correspondence between solutions of S and solutions of C.
    """
    # if k not provided, make it equal to 10 times the max value:
    if k is None:
        # k = round(10*matrix.max())
        k = 99
        
    matrix_bar = matrix.copy()
    np.fill_diagonal(matrix_bar, 0)
    u = np.matrix(np.ones(matrix.shape).astype(int) * k)
    np.fill_diagonal(u, 0)
    matrix_symm_top = np.concatenate((u, np.transpose(matrix_bar)), axis=1)
    matrix_symm_bottom = np.concatenate((matrix_bar, u), axis=1)
    matrix_symm = np.concatenate((matrix_symm_top, matrix_symm_bottom), axis=0)
    
    return matrix_symm


def concorde_tsp_solver(distance_matrix: np.array, node_list: list, specify_end = False):
    """ Solve open TSP problem using concorde. Better path will be selecetd from both directions."""
    distance_matrix_int = np.round(distance_matrix * 5).astype(int)
    k = 10 * np.max(distance_matrix_int)
    if specify_end:
        distance_matrix_int[1:-1, 0] = k  # Specify the starting and ending vertex
    else:
        distance_matrix_int[1:, 0] = k  # So that the staring point will be the first vertex
    
    symm_distance = symmetricize(distance_matrix_int, k = k)
    problem = Problem.from_matrix(symm_distance)
    solution = run_concorde(problem)
    path_indices_with_ghost = solution.tour
    path_indices = path_indices_with_ghost[::2]   # Extract real path index
    path_length = 0 
    for i in range(len(path_indices) - 1):
        start_idx, end_idx = path_indices[i], path_indices[i+1]
        path_length += distance_matrix[start_idx][end_idx]
    ret_indices = path_indices
    ret_length = path_length

    # May be the case that we have two paths: [a0, a1, ..., an] and [a0, an, an-1, ..., a1]
    # Check which is better, and set it as the real tsp path
    # if len(path_indices) >= 3 and not specify_end:   # If specify_end, then do not need to reverse
    if len(path_indices) >= 3:
        candidate_indices = [path_indices[0]] + path_indices[::-1]
        candidate_indices.pop()
        # print(ret_indices)
        # print(candidate_indices)
        # Compare the distance of the two candidate paths using original distance matrix
        candidate_length = 0
        for i in range(len(candidate_indices) - 1):
            start_idx, end_idx = candidate_indices[i], candidate_indices[i+1]
            candidate_length += distance_matrix[start_idx][end_idx]
        if candidate_length < path_length:
            print("Reverse tsp path is better!")
            ret_indices = candidate_indices
            ret_length = candidate_length
        else:
            print("Original tsp path is better!")

    tsp_path = []
    for idx in ret_indices:
        tsp_path.append(node_list[idx])
    return tsp_path, ret_length

if __name__ == "__main__":
    # Construct graph
    random.seed(101)

    node_num = 20
    show_graph = False
    ratio_remove = 0.1
    prior_graph = nx.grid_2d_graph(node_num, node_num)
    # add_weight_attr_to_graph(gridmap, add_random=True)
    add_position_attr_to_graph(prior_graph, add_random=True)
    add_weight_attr_to_graph(prior_graph, add_random=False)
    
    random_remove_nodes(prior_graph, num=int(node_num*node_num*ratio_remove))
    random_remove_edges(prior_graph, num=int(len(prior_graph.edges()) * ratio_remove))

    ## Add covariance for edges
    Cov = np.zeros((3, 3))
    Cov[0, 0] = 0.1
    Cov[1, 1] = 0.1
    Cov[2, 2] = 0.001
    Sigma = np.linalg.inv(Cov)
    add_edge_information_matrix(prior_graph, Sigma)
    add_graph_weights_as_dopt(prior_graph, key="d_opt" )

    node_list = list(prior_graph.nodes())

    distance_matrix = get_distance_matrix_for_new_tsp_solver(prior_graph,\
                                                             node_list=node_list)
    time1 = time.time()
    new_tsp_path, new_tsp_distance = concorde_tsp_solver(distance_matrix, node_list)
    tsp_solve_time = time.time() - time1
    full_tsp_path = connect_tsp_path(prior_graph, new_tsp_path)

    plot_modified_path(prior_graph, full_tsp_path)


    ## Find intial TSP path