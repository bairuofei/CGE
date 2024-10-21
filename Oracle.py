"""This file defines Oracle class that return the contribution of adding edges to graph Laplacian.
"""

import networkx as nx
import numpy as np
import copy
import scipy

from utils.constants import EIG_TH
from utils.utils_matrix import get_d_opt

class Oracle:
    """ A oracle function that evaluates the effect of adding edges to graph Laplacian"""
    def __init__(self, alpha = 1):
        self.alpha = alpha  # Balance graph connectivity and distance metric

    def __get_normalized_log_det_Laplacian(self, reduced_laplacian):
        """ Return 1/n * log det (L) """
        eigv2 = scipy.linalg.eigvalsh(reduced_laplacian)
        if np.iscomplex(eigv2.any()):
            print("Error: Complex Root")
        n = np.size(reduced_laplacian, 1)
        eigv = eigv2[eigv2 > EIG_TH] # Only select eigenvalues larger than EIG_TH (1e-6)
        if len(eigv) < len(eigv2):
            print("Small eigenvalue of graph Laplacian removed.")
        return np.sum(np.log(eigv)) / n

    def set_base_graph(self, base_graph: nx.Graph):
        """Set the prior graph and compute edge length for a connected graph.
        The prior graph includes the following properties:
            1. "position" (node), record the (x, y) coordinate of a node;
            2. "weight" (edge), the distance of edges in base_graph
            3. "information" (edge), the information matrix of a measurement
            4. "d_opt" (edge), the det(information)
        """
        self.base_graph = base_graph
        self.all_edge_length = dict(nx.all_pairs_dijkstra_path_length(self.base_graph, cutoff=None, weight='distance'))
        max_dist = 0
        for edge in self.base_graph.edges():
            max_dist += self.base_graph.edges()[edge]["distance"]
        self.dist_offset = (self.base_graph.number_of_nodes()) ** 2 * max_dist

        Cov = np.zeros((3, 3))
        Cov[0, 0] = 0.1
        Cov[1, 1] = 0.1
        Cov[2, 2] = 0.001
        Sigma = np.linalg.inv(Cov)
        self.gamma = get_d_opt(Sigma)

    def get_dist_offset(self):
        return self.dist_offset
    
    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_initial_graph(self, initial_graph: nx.Graph, node_list: list):
        """ Set the initial base graph. The selected edges will added into the initial graph"""
        self.initial_graph = initial_graph
        self.num_node = self.initial_graph.number_of_nodes()
        self.incidence_vector = np.zeros((self.num_node, 1))

        self.node_list = node_list
        self.node_to_idx = {}
        for idx, node in enumerate(self.node_list):
            self.node_to_idx[node] = idx

        self.laplacian_initial = nx.laplacian_matrix(self.initial_graph, self.node_list, weight="weight").toarray()
        self.laplacian_initial_inverse = np.linalg.inv(self.laplacian_initial[1:, 1:])
    
    def check_inverse(self, edge):
        """Note use matrix determinant lemma and use matrix determinant, has numerical errors"""
        idx1 = self.node_to_idx[edge[0]]
        idx2 = self.node_to_idx[edge[1]]
        self.incidence_vector[idx1] = 1
        self.incidence_vector[idx2] = -1
        n = np.size(self.incidence_vector, 0) - 1
        print(f"Size: {n}")
        # Use matrix determinant lemma
        
        value = np.dot(np.dot(self.incidence_vector[1:, :].T, self.laplacian_initial_inverse),\
                                                        self.incidence_vector[1:, :])[0, 0]
        reduced_laplacian = self.laplacian_initial[1:, 1:]
        gain_laplacian_initial = self.__get_normalized_log_det_Laplacian(reduced_laplacian)
        gain1 = gain_laplacian_initial + np.log(1 + self.gamma * value) / n

        # Use total inverse
        laplacian_update = self.laplacian_initial
        laplacian_update += self.gamma * np.dot(self.incidence_vector, self.incidence_vector.T)
        gain2 = self.__get_normalized_log_det_Laplacian(laplacian_update[1:, 1:])

        self.incidence_vector[idx1] = 0
        self.incidence_vector[idx2] = 0
        distance = self.alpha * 2 * self.all_edge_length[edge[0]][edge[1]]
        print(f"gain1: {gain1}, gain2: {gain2}, distance: {distance}")
        return
    
    def test_combined(self, edges: list):
        """Return f({e}), where {e} is a set of selected edges"""
        # f(G') = 1/n*log(det(L(G'))) - alpha*2*\sum{d(e)}, for all e in G'\G
        # f(G+{e}) - f(G) = marginal_gain
        laplacian_update = copy.copy(self.laplacian_initial)
        gain_distance = 0
        for edge in edges:
            idx1 = self.node_to_idx[edge[0]]
            idx2 = self.node_to_idx[edge[1]]
            self.incidence_vector[idx1] = 1
            self.incidence_vector[idx2] = -1
            laplacian_update += self.gamma * np.dot(self.incidence_vector, self.incidence_vector.T)
            self.incidence_vector[idx1] = 0
            self.incidence_vector[idx2] = 0
            gain_distance += self.all_edge_length[edge[0]][edge[1]]
        reduced_laplacian = laplacian_update[1:, 1:]
        gain_laplacian_set = self.__get_normalized_log_det_Laplacian(reduced_laplacian)

        reduced_laplacian = self.laplacian_initial[1:, 1:]
        gain_initial_laplacian = self.__get_normalized_log_det_Laplacian(reduced_laplacian)
        print(f"After add {len(edges)} edges: ")
        print(f"graph gain: {gain_laplacian_set - gain_initial_laplacian}, distance cost: {self.alpha * 2 * gain_distance}, total: {gain_laplacian_set - self.alpha * 2 * gain_distance}")
        # 1/n log X  ==> X^{1/n}x
        return gain_laplacian_set, self.alpha * 2 * gain_distance


    def gain_combined(self, edges: list, print_gain: bool = False):
        """Return f({e}), where {e} is a set of selected edges"""
        # f(G') = 1/n*log(det(L(G'))) - alpha*2*\sum{d(e)}, for all e in G'\G
        # f(G+{e}) - f(G) = marginal_gain
        laplacian_update = copy.copy(self.laplacian_initial)
        gain_distance = 0
        for edge in edges:
            idx1 = self.node_to_idx[edge[0]]
            idx2 = self.node_to_idx[edge[1]]
            self.incidence_vector[idx1] = 1
            self.incidence_vector[idx2] = -1
            laplacian_update += self.gamma * np.dot(self.incidence_vector, self.incidence_vector.T)
            self.incidence_vector[idx1] = 0
            self.incidence_vector[idx2] = 0
            gain_distance += self.all_edge_length[edge[0]][edge[1]]
        reduced_laplacian = laplacian_update[1:, 1:]
        gain_laplacian_set = self.__get_normalized_log_det_Laplacian(reduced_laplacian)

        reduced_laplacian = self.laplacian_initial[1:, 1:]
        gain_initial_laplacian = self.__get_normalized_log_det_Laplacian(reduced_laplacian)
        if print_gain:
            print(f"After add {len(edges)} edges: ")
            print(f"graph gain: {gain_laplacian_set - gain_initial_laplacian}, \
                  distance cost: {self.alpha * 2 * gain_distance}, \
                    total: {gain_laplacian_set - self.alpha * 2 * gain_distance}")
        return gain_laplacian_set - self.alpha * 2 * gain_distance + self.dist_offset
    
    def ratio_combined(self, edges: list, print_gain: bool = False):
        """Return ratio of laplacian gain and distance cost"""
        laplacian_update = copy.copy(self.laplacian_initial)
        gain_distance = 0
        for edge in edges:
            idx1 = self.node_to_idx[edge[0]]
            idx2 = self.node_to_idx[edge[1]]
            self.incidence_vector[idx1] = 1
            self.incidence_vector[idx2] = -1
            laplacian_update += self.gamma * np.dot(self.incidence_vector, self.incidence_vector.T)
            self.incidence_vector[idx1] = 0
            self.incidence_vector[idx2] = 0
            gain_distance += self.all_edge_length[edge[0]][edge[1]]
        reduced_laplacian = laplacian_update[1:, 1:]
        gain_laplacian_set = self.__get_normalized_log_det_Laplacian(reduced_laplacian)

        reduced_laplacian = self.laplacian_initial[1:, 1:]
        gain_initial_laplacian = self.__get_normalized_log_det_Laplacian(reduced_laplacian)
        return (gain_laplacian_set - gain_initial_laplacian) / (2 * gain_distance), \
            gain_laplacian_set - gain_initial_laplacian, 2 * gain_distance


    def gain_individual(self, edge: tuple):
        """Return f(e), where e is a single edge"""
        # f(e) = log(det(L(G+e))) - alph*2*w(e)
        reduced_laplacian = self.laplacian_initial[1:, 1:]
        gain_laplacian_initial = self.__get_normalized_log_det_Laplacian(reduced_laplacian)
        idx1 = self.node_to_idx[edge[0]]
        idx2 = self.node_to_idx[edge[1]]
        self.incidence_vector[idx1] = 1
        self.incidence_vector[idx2] = -1
        value = np.dot(np.dot(self.incidence_vector[1:, :].T, self.laplacian_initial_inverse),\
                                                        self.incidence_vector[1:, :])[0, 0]
        gain_laplacian_marginal = np.log(1 + self.gamma * value)
        n = np.size(reduced_laplacian, 1)
        gain_laplacian = gain_laplacian_initial + gain_laplacian_marginal / n
        ## Compute the distance over graph, because the edge may not be directly connected
        gain_distance = self.all_edge_length[edge[0]][edge[1]]
        self.incidence_vector[idx1] = 0
        self.incidence_vector[idx2] = 0
        # print(f"single laplacian: {gain_laplacian}")
        return gain_laplacian - self.alpha * 2 * gain_distance + self.dist_offset

