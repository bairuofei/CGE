"""This file defines a TopoGraph class that generates random topo-metric graphs to explore."""

import math
import sys
import random
import networkx as nx
import numpy as np

from utils.utils_graph import get_d_opt

class TopoGraph:
    """A prior topological graph has following properties:
        1. "position" (node), record the (x, y) coordinate of a node;
        2. "distance" (edge), the distance between two nodes
        3. "information" (edge), the information matrix of a measurement
        4. "weight" (edge), the det(information)
    """
    def __init__(self):
        return
    
    def grid_2d_graph(self, node_num1: int, node_num2:int):
        self.graph = nx.grid_2d_graph(node_num1, node_num2)
        return
    
    def map_name_to_str(self):
        """ Change node name from tuple to string. """
        mapping = {(i, j): f'({i}, {j})' for i, j in self.graph.nodes()}
        self.graph = nx.relabel_nodes(self.graph, mapping, copy=False)
        return

    def add_node_position(self, add_random=False):
        """ Only applicable for nodes with tuple (x, y) as their names"""
        for node in self.graph.nodes():
            if add_random:
                variance = 0.2
                new_pos = (node[0] + np.random.normal(0, variance), node[1] + np.random.normal(0, variance))
                self.graph.nodes()[node]["position"] = new_pos
            else:
                self.graph.nodes()[node]["position"] = node
        return
    
    def add_edge_distance(self):
        """ Add edge 'distance' attribute. """
        for edge in self.graph.edges():
            node1, node2 = edge
            pose1, pose2 = self.graph.nodes()[node1]["position"],\
                           self.graph.nodes()[node2]["position"]
            dist = math.sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2)
            self.graph.edges()[edge]["distance"] = dist
        return
    
    def add_edge_information_matrix(self, Sigma: np.ndarray):
        for edge in self.graph.edges():
            self.graph.edges()[edge]["information"] = Sigma
        return
    
    def add_edge_weight(self, key="weight"):
        for edge in self.graph.edges():
            dopt = get_d_opt(self.graph.edges[edge]["information"])
            self.graph.edges[edge][key] = dopt # * random.randint(1, 3)
        return
    
    def random_remove_nodes(self, num = 5):
        """ Remove num of nodes from graph while keeping graph connected. """
        node_list = list(self.graph.nodes())
        if len(node_list) < 2*num:
            print("Very few nodes in graph, cannot remove.")
            sys.exit(1)
        removed = set()
        while num != 0:
            idx = random.randint(0, len(node_list) - 1)
            if idx in removed:
                continue
            copy_graph = self.graph.copy()
            copy_graph.remove_node(node_list[idx])
            if nx.is_connected(copy_graph):
                self.graph.remove_node(node_list[idx])
                num -= 1
                removed.add(idx)
        return 

    def random_remove_edges(self, num=5):
        edge_list = list(self.graph.edges())
        if len(edge_list) < 2 * num:
            print("Very few edges in graph, cannot remove.")
            sys.exit(1)
        removed = set()
        while num != 0:
            idx = random.randint(0, len(edge_list) - 1)
            if idx in removed:
                continue
            copy_graph = self.graph.copy()
            copy_graph.remove_edge(edge_list[idx][0], edge_list[idx][1])
            if nx.is_connected(copy_graph):
                self.graph.remove_edge(edge_list[idx][0], edge_list[idx][1])
                num -= 1
                removed.add(idx)
        return 


    def get_node_list(self) -> list:
        return list(self.graph.nodes())
    

    def get_distance_matrix(self, node_list: list, use_dijsktra_edge: bool=True, attr="distance", scale_to_int = False) -> np.ndarray:
        """ Return a symmetric distance matrix for vertex in node_list, maintaining its order.
        sclae_to_int means the distance times 100
        """
        all_length = dict(nx.all_pairs_dijkstra_path_length(self.graph, cutoff=None, weight=attr))
        dist_matrix = np.zeros((len(node_list), len(node_list)))
        for i, curr in enumerate(node_list):
            for j in range(i):
                prev = node_list[j]          
                if use_dijsktra_edge:  # To effectively consider the distance
                    dist = all_length[prev][curr]
                else:
                    if ((prev, curr) in self.graph.edges()) or ((curr, prev) in self.graph.edges()):
                        dist = self.graph.edges()[(prev, curr)][attr]
                    else:
                        dist = float("inf")
                dist_matrix[i][j] = dist
        symmetric_matrix = dist_matrix + dist_matrix.T

        if scale_to_int:
            symmetric_matrix_int = np.round(symmetric_matrix * 100).astype(int)
            return symmetric_matrix_int

        return symmetric_matrix
    
    def get_path_length(self, path: list, attr="distance"):
        dist = 0
        for i in range(len(path) - 1):
            try:
                dist += nx.shortest_path_length(self.graph, source=path[i], target=path[i+1], \
                                                weight=attr, method='dijkstra')
            except nx.NodeNotFound or nx.NetworkXNoPath or nx.ValueError:
                return -1
        return dist
    
    def connect_tsp_path(self, tsp_path: list, attr="distance") -> list:
        """ Connect disjoint nodes in TSP path with shortest path"""
        path = []
        for i in range(len(tsp_path) - 1):
            curr = tsp_path[i]
            next = tsp_path[i+1]
            if (curr, next) in self.graph.edges() or (next, curr) in self.graph.edges():
                path.append(curr)
            else:
                try:
                    connect_path = nx.shortest_path(self.graph, source=curr, target=next, weight=attr, method='dijkstra')
                    path += connect_path[:-1]
                except nx.NodeNotFound or nx.NetworkXNoPath or nx.ValueError:
                    print("Cannot connect tsp_path!")
        path.append(tsp_path[-1])
        return path
