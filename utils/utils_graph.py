import math
import random
import sys
import scipy
import numpy as np
import networkx as nx
from typing import Tuple, List

from utils.utils_matrix import get_d_opt

from utils.constants import EIG_TH

def add_weight_attr_to_graph(graph):
    for edge in graph.edges():
        node1, node2 = edge
        pose1, pose2 = graph.nodes()[node1]["position"], graph.nodes()[node2]["position"]
        dist = math.sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2)
        graph.edges()[edge]["weight"] = dist


def add_edge_information_matrix(graph, Sigma):
    for edge in graph.edges():
        graph.edges()[edge]["information"] = Sigma


def add_graph_weights_as_dopt(graph, key="weight"):
    for edge in graph.edges():
        dopt = get_d_opt(graph.edges[edge]["information"])
        graph.edges[edge][key] = dopt # * random.randint(1, 3)
    return

def add_position_attr_to_graph(graph, add_random=False):
    """ Only applicable for nodes with tuple (x, y) as their names"""
    for node in graph.nodes():
        if add_random:
            variance = 0.2
            new_pos = (node[0] + np.random.normal(0, variance), node[1] + np.random.normal(0, variance))
            graph.nodes()[node]["position"] = new_pos
        else:
            graph.nodes()[node]["position"] = node


def generate_random_gridmap(size_x, size_y, covariance = None):
    size_x, size_y = 6, 6
    gridmap = nx.grid_2d_graph(size_x, size_y)
    add_position_attr_to_graph(gridmap, scale = 12)
    add_weight_attr_to_graph(gridmap)

    ## TODO: covariance estimation based on vertex observability
    if covariance is None:
        Cov = np.zeros((3, 3))
        Cov[0, 0] = 0.1
        Cov[1, 1] = 0.1
        Cov[2, 2] = 0.001
    else:
        Cov = covariance
    Sigma = np.linalg.inv(Cov)
    add_edge_information_matrix(gridmap, Sigma)
    add_graph_weights_as_dopt(gridmap, key="d_opt")
    return gridmap

def random_remove_nodes(graph, num = 5):
    """ Remove num of nodes from graph while keeping graph connected. """
    node_list = list(graph.nodes())
    if len(node_list) < 2*num:
        print("Very few nodes in graph, cannot remove.")
        sys.exit(1)
    removed = set()
    while num != 0:
        idx = random.randint(0, len(node_list) - 1)
        if idx in removed:
            continue
        copy_graph = graph.copy()
        copy_graph.remove_node(node_list[idx])
        if nx.is_connected(copy_graph):
            graph.remove_node(node_list[idx])
            num -= 1
            removed.add(idx)
    return 

def random_remove_edges(graph, num=5):
    edge_list = list(graph.edges())
    if len(edge_list) < 2 * num:
        print("Very few edges in graph, cannot remove.")
        sys.exit(1)
    removed = set()
    while num != 0:
        idx = random.randint(0, len(edge_list) - 1)
        if idx in removed:
            continue
        copy_graph = graph.copy()
        copy_graph.remove_edge(edge_list[idx][0], edge_list[idx][1])
        if nx.is_connected(copy_graph):
            graph.remove_edge(edge_list[idx][0], edge_list[idx][1])
            num -= 1
            removed.add(idx)
    return 


def get_path_length(graph, path, weight="weight"):
    dist = 0
    for i in range(len(path) - 1):
        try:
            dist += nx.shortest_path_length(graph, source=path[i], target=path[i+1], weight=weight, method='dijkstra')
        except nx.NodeNotFound or nx.NetworkXNoPath or nx.ValueError:
            return -1
    return dist

def copy_node_with_new_name(G: nx.graph, node: str, new_name: str, edge_attrs: dict) -> nx.graph:
    """ Copy a node in a graph, which is only different in name"""
    G.add_node(new_name)
    
    # Copy attributes from the original node to the new node
    G.nodes[new_name].update(G.nodes[node])
    
    # Copy connections (edges) from the original node to the new node
    neighbors = G.edges(node, data=True)
    for node1, node2, attrs in neighbors:
        neighbor = node1
        if node1 == node:
            neighbor = node2
        G.add_edge(new_name, neighbor, **attrs)
    return G


# TSP Planner
def get_distance_matrix_for_tsp(graph: nx.graph, use_dijsktra_edge=True) -> Tuple[list, list]:
    """Given a networkx graph object, return the distance matrix for tsp-solver.
    node_list is also returned to provide index of nodes in tsp-solver.
    D = [[],
         [0-1],
         [0-2, 1-2],
         [0-3, 1-3, 2-3],
         ...]
    """
    node_list = list(graph.nodes())
    all_length = dict(nx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='weight'))
    D = []
    for i, curr in enumerate(node_list):
        D.append([])
        for j in range(i):
            prev = node_list[j]          
            if use_dijsktra_edge:  # To effectively consider the distance
                dist = all_length[prev][curr]
            else:
                if ((prev, curr) in graph.edges()) or ((curr, prev) in graph.edges()):
                    dist = graph.edges()[(prev, curr)]["weight"]
                else:
                    dist = float("inf")
            D[-1].append(dist)
    return D, node_list


def get_distance_matrix_for_new_tsp_solver(graph: nx.graph, node_list: list, use_dijsktra_edge=True) -> np.array:
    """ Return a symmetric distance matrix for vertex in node_list, maintaining its order.
    """
    all_length = dict(nx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='weight'))
    dist_matrix = np.zeros((len(node_list), len(node_list)))
    for i, curr in enumerate(node_list):
        for j in range(i):
            prev = node_list[j]          
            if use_dijsktra_edge:  # To effectively consider the distance
                dist = all_length[prev][curr]
            else:
                if ((prev, curr) in graph.edges()) or ((curr, prev) in graph.edges()):
                    dist = graph.edges()[(prev, curr)]["weight"]
                else:
                    dist = float("inf")
            dist_matrix[i][j] = dist
    symmetric_matrix = dist_matrix + dist_matrix.T
    return symmetric_matrix

def connect_tsp_path(graph, tsp_path):
    """ The TSP path has some vertices not connected directly. Connect these vertices over g_prior """
    path = []
    for i in range(len(tsp_path) - 1):
        curr = tsp_path[i]
        next = tsp_path[i+1]
        if (curr, next) in graph.edges() or (next, curr) in graph.edges():
            path.append(curr)
        else:
            try:
                connect_path = nx.shortest_path(graph, source=curr, target=next, weight="weight", method='dijkstra')
                # print(curr)
                # print(next)
                # print(connect_path)
                path += connect_path[:-1]
            except nx.NodeNotFound or nx.NetworkXNoPath or nx.ValueError:
                print("Cannot connect tsp_path!")
    path.append(tsp_path[-1])
    return path


## Graph Laplacian
def get_normalized_weighted_spanning_trees(graph: nx.Graph, weight: str="weight"):
    """Return normalized determinant of reduced Laplacian"""
    laplacian = nx.laplacian_matrix(graph, weight=weight).toarray()
    reduced_laplacian = laplacian[1:, 1:]
    eigv2 = scipy.linalg.eigvalsh(reduced_laplacian)
    if np.iscomplex(eigv2.any()):
        print("Error: Complex Root")
    n = np.size(reduced_laplacian, 1)
    eigv = eigv2[eigv2 > EIG_TH] # Only select eigenvalues larger than EIG_TH (1e-6)
    return np.exp(np.sum(np.log(eigv)) / n)


## G2O graph
def write_nx_graph_to_g2o(graph: nx.graph, save_path: str, nodes_start: list = []):
    """ Nodes in nodes_start will be moved to the front
    """
    all_nodes_origin = list(graph.nodes())
    all_nodes = []
    for node in nodes_start:
        all_nodes.append(node)
    for node in all_nodes_origin:
        if node not in nodes_start:
            all_nodes.append(node)
    # TODO: Move starting vertex as the robots' starting vertices
    node_to_idx = {all_nodes[idx]: idx for idx in range(len(all_nodes))}
    with open(save_path, 'w') as file: # w is write mode, file content will be clear. Use a if want to append
        for i in range(len(all_nodes)):
            node = all_nodes[i]
            curr_line = "VERTEX_SE2 "
            curr_line += str(i)
            for item in graph.nodes()[node]["position"]:
                curr_line += " " + str(item)
            curr_line += " 0"  # Because 2D grid does not provide theta angle
            curr_line += "\n"
            file.write(curr_line)
        for edge in graph.edges():
            curr_line = "EDGE_SE2 "
            curr_line += str(node_to_idx[edge[0]]) + " " + str(node_to_idx[edge[1]]) + " "
            information = graph.edges()[edge]["information"]
            cov = np.linalg.inv(information)
            curr_line += str(cov[0, 0]) + " " + str(cov[0, 1]) + " " + str(cov[0, 2])\
                        + " " + str(cov[1, 1]) + " " + str(cov[1, 2]) + " " \
                        + str(cov[2, 2]) + "\n"
            file.write(curr_line)
    return