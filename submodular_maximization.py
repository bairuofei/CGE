"""This file implements several algorithms to solve the loop edge selection problem. 
Including simple greedy; double greedy and deterministicUSM (with 1/2 optimality guarantee).

The executed code aims to test the function using a single-robot case.
"""



import random
import time as mytime
import numpy as np
from collections import defaultdict
import copy
import heapq
from typing import Tuple

from pulp import *

from utils.utils_tsp import concorde_tsp_solver

from Oracle import Oracle
from TopoGraph import TopoGraph


def iterative_greedy(ground_set: list, oracle: Oracle) -> Tuple[set, float]:
    print("Call iterative_greedy...")
    curr_obj = oracle.gain_combined([])
    curr_selected = []
    valid_indices = set([i for i in range(len(ground_set))])
    while True:
        gain_record = []
        for k in valid_indices:
            combined_edges = curr_selected + [ground_set[k]]
            gain_record.append([oracle.gain_combined(combined_edges), k])

        gain_record.sort(reverse=True)
        valid_indices.remove(gain_record[0][1])
        if gain_record[0][0] > curr_obj:
            curr_selected.append(ground_set[gain_record[0][1]])
            curr_obj = oracle.gain_combined(curr_selected)
        else:
            break
        if len(valid_indices) == 0:
            break
    
    return set(curr_selected), curr_obj

def iterative_greedy_heap(ground_set: list, oracle: Oracle) -> Tuple[set, float]:
    print("Call iterative_greedy...")
    
    valid_indices = set([i for i in range(len(ground_set))])

    # Maintain a heap, with elements [obj, index]
    # Note: only the increase term has submodular property
    heap = []
    initial_obj = oracle.gain_combined([])
    for index in valid_indices:
        obj = oracle.gain_combined([ground_set[index]])
        heapq.heappush(heap, [-(obj - initial_obj), index])

    curr_selected = []
    curr_obj = oracle.gain_combined([])
    while True:
        if len(heap) == 0:
            break
        top = heapq.heappop(heap)
        if len(heap) == 0:
            curr_element = ground_set[top[1]]
        else:
            # Update obj of the top element
            while True:
                combined_edges = curr_selected + [ground_set[top[1]]]
                new_obj = oracle.gain_combined(combined_edges)
                curr_top = heap[0]
                if new_obj - curr_obj >= -curr_top[0]:
                    curr_element = ground_set[top[1]]
                    break
                else:
                    heapq.heappush(heap, [-(new_obj - curr_obj), top[1]])
                    top = heapq.heappop(heap)

        combined_edges = curr_selected + [curr_element]
        this_obj = oracle.gain_combined(combined_edges)
        if this_obj > curr_obj:
            curr_obj = this_obj
            curr_selected.append(curr_element)
        else:
            break
    
    return set(curr_selected), curr_obj


def double_greedy(ground_set: list, oracle: Oracle) -> Tuple[set, float]:
    """Randomized double greedy algorithm to maximize a non-monotone submodular function."""
    print("Call double_greedy..")
    X = set()
    Y = set(ground_set)
    for i in range(len(ground_set)):
        if len(ground_set) > 50 and i % (len(ground_set)//10) == 0:
            print(f"    {i} / {len(ground_set)}")
        new_X = copy.copy(X)
        new_X.add(ground_set[i])
        new_Y = copy.copy(Y)
        new_Y.discard(ground_set[i])
        ai = max(0, oracle.gain_combined(list(new_X)) - oracle.gain_combined(list(X)))
        bi = max(0, oracle.gain_combined(list(new_Y)) - oracle.gain_combined(list(Y)))
        if ai == 0 and bi == 0:
            threshold = 1
        else:
            threshold = ai / (ai + bi)
        sample = random.random()
        if sample <= threshold:
            X = new_X
        else:
            Y = new_Y
    best_value = oracle.gain_combined(list(X))
    return X, best_value


def double_greedy_with_ordering(ground_set: list, oracle: Oracle) -> Tuple[set, float]:
    """Randomized double greedy algorithm to maximize a non-monotone submodular function."""
    print("Call double_greedy_with_ordering...")
    X = set()
    Y = set(ground_set)
    # Get next element by ordering
    valid_indices = set([i for i in range(len(ground_set))])

    while len(valid_indices) > 0:
        if len(ground_set) > 50 and len(valid_indices) % (len(ground_set)//10) == 0:
            print(f"    {len(ground_set) - len(valid_indices)} / {len(ground_set)}")

        gain_record = []
        for k in valid_indices:
            combined_edges = list(X) + [ground_set[k]]
            gain_record.append([oracle.gain_combined(combined_edges), k])
        gain_record.sort(reverse=True)

        valid_indices.remove(gain_record[0][1])
        curr_element = ground_set[gain_record[0][1]]

        new_X = copy.copy(X)
        new_X.add(curr_element)
        new_Y = copy.copy(Y)
        new_Y.discard(curr_element)
        ai = max(0, oracle.gain_combined(list(new_X)) - oracle.gain_combined(list(X)))
        bi = max(0, oracle.gain_combined(list(new_Y)) - oracle.gain_combined(list(Y)))
        if ai == 0 and bi == 0:
            threshold = 1
        else:
            threshold = ai / (ai + bi)
        sample = random.random()
        if sample <= threshold:
            X = new_X
        else:
            Y = new_Y
    best_value = oracle.gain_combined(list(X))
    return X, best_value


def double_greedy_with_ordering_heap(ground_set: list, oracle: Oracle) -> Tuple[set, float]:
    """Randomized double greedy algorithm to maximize a non-monotone submodular function."""
    print("Call double_greedy_with_ordering_heap...")
    X = set()
    Y = set(ground_set)
    # Get next element by ordering
    valid_indices = set([i for i in range(len(ground_set))])

    # Maintain a heap, with elements [obj, index]
    heap = []
    initial_obj = oracle.gain_combined([])
    for index in valid_indices:
        obj = oracle.gain_combined([ground_set[index]])
        heapq.heappush(heap, [-(obj - initial_obj), index])

    while len(heap) > 0:
        if len(ground_set) > 50 and len(valid_indices) % (len(ground_set)//10) == 0:
            print(f"    {len(ground_set) - len(valid_indices)} / {len(ground_set)}")

        top = heapq.heappop(heap)
        if len(heap) == 0:
            curr_element = ground_set[top[1]]
        else:
            # Update obj of the top element
            curr_obj_X = oracle.gain_combined(list(X))
            while True:
                combined_edges = list(X) + [ground_set[top[1]]]
                new_obj = oracle.gain_combined(combined_edges)
                curr_top = heap[0]
                if new_obj - curr_obj_X >= -curr_top[0]:
                    curr_element = ground_set[top[1]]
                    break
                else:
                    heapq.heappush(heap, [-(new_obj - curr_obj_X), top[1]])
                    top = heapq.heappop(heap)

        new_X = copy.copy(X)
        new_X.add(curr_element)
        new_Y = copy.copy(Y)
        new_Y.discard(curr_element)
        ai = max(0, oracle.gain_combined(list(new_X)) - oracle.gain_combined(list(X)))
        bi = max(0, oracle.gain_combined(list(new_Y)) - oracle.gain_combined(list(Y)))
        if ai == 0 and bi == 0:
            threshold = 1
        else:
            threshold = ai / (ai + bi)
        sample = random.random()
        if sample <= threshold:
            X = new_X
        else:
            Y = new_Y
    best_value = oracle.gain_combined(list(X))
    return X, best_value



def deterministicUSM(ground_set: list, oracle: Oracle) -> Tuple[frozenset, float]:
    """ Find the best set to maixmize a non-monotone submodular function.
    However, if the minimum distance cost is larger than information increase, the objective function 
    is monotonically decreasing as more edges are selected, in which case an optimal solution is an empty set.
    """
    print("Call deterministicUSM...")
    belief = defaultdict(float)
    # key: (set X, set Y), value: probability
    belief[(frozenset(), frozenset(ground_set))] = 1
    for i in range(len(ground_set)):
        if len(ground_set) > 50 and i % (len(ground_set)//10) == 0:
            print(f"    {i} / {len(ground_set)}")
        a_X = []
        b_Y = []
        z_XY = []
        w_XY = []
        key_order = list(belief.keys())
        for idx, key in enumerate(key_order):
            X, Y = key
            X_plus = set(X)
            X_plus.add(ground_set[i])
            a_X.append(oracle.gain_combined(list(X_plus)) - oracle.gain_combined(list(X)))

            Y_minus = set(Y)
            Y_minus.discard(ground_set[i]) # better than remove() method
            b_Y.append(oracle.gain_combined(list(Y_minus)) - oracle.gain_combined(list(Y)))

            z_XY.append(LpVariable("z" + str(idx), 0, 1))
            w_XY.append(LpVariable("w" + str(idx), 0, 1))
        # print("a_X: ")
        # print(a_X)
        # print("b_Y: ")
        # print(b_Y)

        # Construct linear program
        prob = LpProblem("myProblem", LpMinimize)
        inequality1_LHS = []
        inequality1_RHS = []
        inequality2_LHS = []  # the same as inequality1_LHS
        inequality2_RHS = []
        for j in range(len(z_XY)):
            # FIXED: Expection should multiply probability
            prob += z_XY[j] + w_XY[j] == 1
            inequality1_LHS.append((z_XY[j] * a_X[j] + w_XY[j] * b_Y[j]) * belief[key_order[j]])
            inequality1_RHS.append(z_XY[j] * b_Y[j] * belief[key_order[j]])
            inequality2_RHS.append(w_XY[j] * a_X[j] * belief[key_order[j]])
        prob += lpSum(inequality1_LHS) - 2 * lpSum(inequality1_RHS) >= 0
        prob += lpSum(inequality1_LHS) - 2 * lpSum(inequality2_RHS) >= 0

        # TODO: Objective function?
        prob += 0.5* lpSum(z_XY) + 0.6 * lpSum(w_XY) 

        # status = prob.solve(logPath="stats.log")
        status = prob.solve(apis.PULP_CBC_CMD(logPath="./stats.log"))
        # for v in prob.variables():
        #     print(v.name, "=", v.varValue)

        # Construct new distribution
        new_belief = defaultdict(float)
        for idx, key in enumerate(key_order):
            X, Y = key
            if value(z_XY[idx]) > 0:  # prefer add
                new_prob = value(z_XY[idx]) * belief[key]
                new_X = set(X)
                new_Y = set(Y)
                new_X.add(ground_set[i])
                frozen_new_X = frozenset(new_X)
                frozen_new_Y = frozenset(new_Y)
                if (frozen_new_X, frozen_new_Y) in new_belief:
                    new_belief[(frozen_new_X, frozen_new_Y)] += new_prob
                else:
                    new_belief[(frozen_new_X, frozen_new_Y)] = new_prob
            if value(w_XY[idx]) > 0:  # prefer remove
                new_prob = value(w_XY[idx]) * belief[key]
                new_X = set(X)
                new_Y = set(Y)
                new_Y.discard(ground_set[i])
                frozen_new_X = frozenset(new_X)
                frozen_new_Y = frozenset(new_Y)
                if (frozen_new_X, frozen_new_Y) in new_belief:
                    new_belief[(frozen_new_X, frozen_new_Y)] += new_prob
                else:
                    new_belief[(frozen_new_X, frozen_new_Y)] = new_prob
        belief = new_belief

    # Return max solution
    best_X = set()
    best_value = None
    for key in belief.keys():
        X, Y = key
        oracle_value = oracle.gain_combined(list(X))
        if best_value is None or best_value < oracle_value:
            best_value = oracle_value
            best_X = X
    return best_X, best_value


def deterministicUSM_with_ordering(ground_set: list, oracle: Oracle) -> Tuple[frozenset, float]:
    """ Find the best set to maixmize a non-monotone submodular function.
    However, if the minimum distance cost is larger than information increase, the objective function 
    is monotonically decreasing as more edges are selected, in which case an optimal solution is the 
    optimal set.
    """
    print("Call deterministicUSM_with_ordering...")
    belief = defaultdict(float)
    # key: (set X, set Y), value: probability
    belief[(frozenset(), frozenset(ground_set))] = 1
    valid_indices = set([i for i in range(len(ground_set))])

    while len(valid_indices) > 0:
        if len(ground_set) > 50 and len(valid_indices) % (len(ground_set)//10) == 0:
            print(f"    {len(ground_set) - len(valid_indices)} / {len(ground_set)}")
        # Give next element according to the largest possible belief
        max_belief_value = 0
        curr_best_belief = set()
        for key, belief_value in belief.items():
            if belief_value > max_belief_value:
                max_belief_value = belief_value
                X, Y = key
                curr_best_belief = set(X)
        
        gain_record = []
        for k in valid_indices:
            combined_edges = list(curr_best_belief) + [ground_set[k]]
            gain_record.append([oracle.gain_combined(combined_edges), k])
        gain_record.sort(reverse=True)

        valid_indices.remove(gain_record[0][1])
        curr_element = ground_set[gain_record[0][1]]

        a_X = []
        b_Y = []
        z_XY = []
        w_XY = []
        key_order = list(belief.keys())
        for idx, key in enumerate(key_order):
            X, Y = key
            X_plus = set(X)
            X_plus.add(curr_element)
            a_X.append(oracle.gain_combined(list(X_plus)) - oracle.gain_combined(list(X)))

            Y_minus = set(Y)
            Y_minus.discard(curr_element) # better than remove() method
            b_Y.append(oracle.gain_combined(list(Y_minus)) - oracle.gain_combined(list(Y)))

            z_XY.append(LpVariable("z" + str(idx), 0, 1))
            w_XY.append(LpVariable("w" + str(idx), 0, 1))
        # print("a_X: ")
        # print(a_X)
        # print("b_Y: ")
        # print(b_Y)

        # Construct linear program
        prob = LpProblem("myProblem", LpMinimize)
        inequality1_LHS = []
        inequality1_RHS = []
        inequality2_LHS = []  # the same as inequality1_LHS
        inequality2_RHS = []
        for j in range(len(z_XY)):
            # FIXED: Expection should multiply probability
            prob += z_XY[j] + w_XY[j] == 1
            inequality1_LHS.append((z_XY[j] * a_X[j] + w_XY[j] * b_Y[j]) * belief[key_order[j]])
            inequality1_RHS.append(z_XY[j] * b_Y[j] * belief[key_order[j]])
            inequality2_RHS.append(w_XY[j] * a_X[j] * belief[key_order[j]])
        prob += lpSum(inequality1_LHS) - 2 * lpSum(inequality1_RHS) >= 0
        prob += lpSum(inequality1_LHS) - 2 * lpSum(inequality2_RHS) >= 0

        # TODO: Objective function?
        prob += 0.5* lpSum(z_XY) + 0.6 * lpSum(w_XY) 

        # status = prob.solve(logPath="stats.log")
        status = prob.solve(apis.PULP_CBC_CMD(logPath="./stats.log"))
        # for v in prob.variables():
        #     print(v.name, "=", v.varValue)

        # Construct new distribution
        new_belief = defaultdict(float)
        for idx, key in enumerate(key_order):
            X, Y = key
            if value(z_XY[idx]) > 0:  # prefer add
                new_prob = value(z_XY[idx]) * belief[key]
                new_X = set(X)
                new_Y = set(Y)
                new_X.add(curr_element)
                frozen_new_X = frozenset(new_X)
                frozen_new_Y = frozenset(new_Y)
                if (frozen_new_X, frozen_new_Y) in new_belief:
                    new_belief[(frozen_new_X, frozen_new_Y)] += new_prob
                else:
                    new_belief[(frozen_new_X, frozen_new_Y)] = new_prob
            if value(w_XY[idx]) > 0:  # prefer remove
                new_prob = value(w_XY[idx]) * belief[key]
                new_X = set(X)
                new_Y = set(Y)
                new_Y.discard(curr_element)
                frozen_new_X = frozenset(new_X)
                frozen_new_Y = frozenset(new_Y)
                if (frozen_new_X, frozen_new_Y) in new_belief:
                    new_belief[(frozen_new_X, frozen_new_Y)] += new_prob
                else:
                    new_belief[(frozen_new_X, frozen_new_Y)] = new_prob
        belief = new_belief

    # Return max solution
    best_X = set()
    best_value = None
    for key in belief.keys():
        X, Y = key
        oracle_value = oracle.gain_combined(list(X))
        if best_value is None or best_value < oracle_value:
            best_value = oracle_value
            best_X = X
    return best_X, best_value


def deterministicUSM_with_ordering_heap(ground_set: list, oracle: Oracle) -> Tuple[frozenset, float]:
    """ Find the best set to maixmize a non-monotone submodular function.
    However, if the minimum distance cost is larger than information increase, the objective function 
    is monotonically decreasing as more edges are selected, in which case an optimal solution is the 
    optimal set.
    """
    print("Call deterministicUSM_with_ordering...")
    belief = defaultdict(float)
    # key: (set X, set Y), value: probability
    belief[(frozenset(), frozenset(ground_set))] = 1
    valid_indices = set([i for i in range(len(ground_set))])

    # Maintain a heap, with elements [obj, index]
    # Note: only the increase term has submodular property
    heap = []
    initial_obj = oracle.gain_combined([])
    for index in valid_indices:
        obj = oracle.gain_combined([ground_set[index]])
        heapq.heappush(heap, [-(obj - initial_obj), index])

    curr_best_belief = set()
    max_belief_value = 0
    while len(heap) > 0:
        if len(ground_set) > 50 and len(valid_indices) % (len(ground_set)//10) == 0:
            print(f"    {len(ground_set) - len(valid_indices)} / {len(ground_set)}")
        
        curr_belief_obj = oracle.gain_combined(list(curr_best_belief))
        top = heapq.heappop(heap)
        if len(heap) == 0:
            curr_element = ground_set[top[1]]
        else:
            # Update obj of the top element
            while True:
                combined_edges = list(curr_best_belief) + [ground_set[top[1]]]
                new_obj = oracle.gain_combined(combined_edges)
                curr_top = heap[0]
                if new_obj - curr_belief_obj >= -curr_top[0]:
                    curr_element = ground_set[top[1]]
                    break
                else:
                    heapq.heappush(heap, [-(new_obj - curr_belief_obj), top[1]])
                    top = heapq.heappop(heap)

        a_X = []
        b_Y = []
        z_XY = []
        w_XY = []
        key_order = list(belief.keys())
        for idx, key in enumerate(key_order):
            X, Y = key
            X_plus = set(X)
            X_plus.add(curr_element)
            a_X.append(oracle.gain_combined(list(X_plus)) - oracle.gain_combined(list(X)))

            Y_minus = set(Y)
            Y_minus.discard(curr_element) # better than remove() method
            b_Y.append(oracle.gain_combined(list(Y_minus)) - oracle.gain_combined(list(Y)))

            z_XY.append(LpVariable("z" + str(idx), 0, 1))
            w_XY.append(LpVariable("w" + str(idx), 0, 1))

        # Construct linear program
        prob = LpProblem("myProblem", LpMinimize)
        inequality1_LHS = []
        inequality1_RHS = []
        inequality2_LHS = []  # the same as inequality1_LHS
        inequality2_RHS = []
        for j in range(len(z_XY)):
            # FIXED: Expection should multiply probability
            prob += z_XY[j] + w_XY[j] == 1
            inequality1_LHS.append((z_XY[j] * a_X[j] + w_XY[j] * b_Y[j]) * belief[key_order[j]])
            inequality1_RHS.append(z_XY[j] * b_Y[j] * belief[key_order[j]])
            inequality2_RHS.append(w_XY[j] * a_X[j] * belief[key_order[j]])
        prob += lpSum(inequality1_LHS) - 2 * lpSum(inequality1_RHS) >= 0
        prob += lpSum(inequality1_LHS) - 2 * lpSum(inequality2_RHS) >= 0

        # TODO: Objective function?
        prob += 0.5* lpSum(z_XY) + 0.6 * lpSum(w_XY) 

        # status = prob.solve(logPath="stats.log")
        status = prob.solve(apis.PULP_CBC_CMD(logPath="./stats.log"))
        # for v in prob.variables():
        #     print(v.name, "=", v.varValue)

        # Construct new distribution
        new_belief = defaultdict(float)
        curr_best_belief = set()
        max_belief_value = 0
        for idx, key in enumerate(key_order):
            X, Y = key
            if value(z_XY[idx]) > 0:  # prefer add
                new_prob = value(z_XY[idx]) * belief[key]
                new_X = set(X)
                new_Y = set(Y)
                new_X.add(curr_element)
                frozen_new_X = frozenset(new_X)
                frozen_new_Y = frozenset(new_Y)
                if (frozen_new_X, frozen_new_Y) in new_belief:
                    new_belief[(frozen_new_X, frozen_new_Y)] += new_prob
                    if new_belief[(frozen_new_X, frozen_new_Y)] > max_belief_value:
                        max_belief_value = new_belief[(frozen_new_X, frozen_new_Y)]
                        curr_best_belief = set(frozen_new_X)
                else:
                    new_belief[(frozen_new_X, frozen_new_Y)] = new_prob
                    if new_belief[(frozen_new_X, frozen_new_Y)] > max_belief_value:
                        max_belief_value = new_belief[(frozen_new_X, frozen_new_Y)]
                        curr_best_belief = set(frozen_new_X)
            if value(w_XY[idx]) > 0:  # prefer remove
                new_prob = value(w_XY[idx]) * belief[key]
                new_X = set(X)
                new_Y = set(Y)
                new_Y.discard(curr_element)
                frozen_new_X = frozenset(new_X)
                frozen_new_Y = frozenset(new_Y)
                if (frozen_new_X, frozen_new_Y) in new_belief:
                    new_belief[(frozen_new_X, frozen_new_Y)] += new_prob
                    if new_belief[(frozen_new_X, frozen_new_Y)] > max_belief_value:
                        max_belief_value = new_belief[(frozen_new_X, frozen_new_Y)]
                        curr_best_belief = set(frozen_new_X)
                else:
                    new_belief[(frozen_new_X, frozen_new_Y)] = new_prob
                    if new_belief[(frozen_new_X, frozen_new_Y)] > max_belief_value:
                        max_belief_value = new_belief[(frozen_new_X, frozen_new_Y)]
                        curr_best_belief = set(frozen_new_X)
        belief = new_belief

    # Return max solution
    best_X = set()
    best_value = None
    for key in belief.keys():
        X, Y = key
        oracle_value = oracle.gain_combined(list(X))
        if best_value is None or best_value < oracle_value:
            best_value = oracle_value
            best_X = X
    return best_X, best_value



if __name__ == "__main__":

    random.seed(101)
    show_graph = False
    alpha = 0.007   # trade-off between distance and graph Laplacian

    node_num = 10
    ratio_remove = 0.05
    add_random_in_position = True
    prior_graph = TopoGraph()
    prior_graph.grid_2d_graph(node_num, node_num)
    prior_graph.add_node_position(add_random=add_random_in_position)
    prior_graph.add_edge_distance()

    prior_graph.random_remove_nodes()
    prior_graph.random_remove_edges()

    ## Add covariance for edges
    Cov = np.zeros((3, 3))
    Cov[0, 0] = 0.1
    Cov[1, 1] = 0.1
    Cov[2, 2] = 0.001
    Sigma = np.linalg.inv(Cov)
    prior_graph.add_edge_information_matrix(Sigma)
    prior_graph.add_edge_weight()

    # Solve TSP path planning
    node_list = prior_graph.get_node_list()
    #TODO: Specify starting node. By default is (0, 0)
    distance_matrix = prior_graph.get_distance_matrix(node_list)
    tsp_path, tsp_distance = concorde_tsp_solver(distance_matrix, node_list)
    # tsp_solve_time = time.time() - time1
    full_tsp_path = prior_graph.connect_tsp_path(tsp_path)


    ## Extract TSP-path induced base graph
    tsp_edges = set()
    for i in range(len(full_tsp_path) - 1):
        tsp_edges.add((full_tsp_path[i], full_tsp_path[i+1]))
        tsp_edges.add((full_tsp_path[i+1], full_tsp_path[i]))
    G_tsp = prior_graph.graph.edge_subgraph(tsp_edges).copy()
    G_tsp_node_list = list(G_tsp.nodes())
    G_tsp_edge_list = set(G_tsp.edges())

    one_edge = list(G_tsp_edge_list)[0]
    print(one_edge)
    print(prior_graph.graph.nodes()[one_edge[0]]["position"])
    print(prior_graph.graph.nodes()[one_edge[1]]["position"])
    print(prior_graph.graph.edges()[one_edge]["weight"])

    # Set Oracle function
    oracle_function = Oracle(alpha=alpha)
    oracle_function.set_base_graph(prior_graph.graph)
    oracle_function.set_initial_graph(G_tsp, G_tsp_node_list)


    valid_closures = set()
    initial_obj_value = oracle_function.gain_combined([])
    # removed_closures = set()
    # Find the maximum marginal increase of D-opt(L), in order to derive radius of KD-Tree
    for i, curr in enumerate(full_tsp_path):
        if i % 100 == 0:
            print(f"current idx [find max threshold]: {i} / {len(full_tsp_path) - 1}")      
        for closure in full_tsp_path[:i]:   # search in its previous vertex
            if (curr, closure) in G_tsp.edges() or (closure, curr) in G_tsp.edges() or curr == closure:
                continue
            if (curr, closure) in valid_closures or (closure, curr) in valid_closures:
                continue  
            curr_loop = (closure, curr) 
            curr_obj_value = oracle_function.gain_combined([curr_loop])
            if curr_obj_value > initial_obj_value:
                valid_closures.add(curr_loop)
            # gain1 = oracle_function.gain_individual(curr_loop)
            # gain = oracle_function.gain_combined([curr_loop])
            # oracle_function.check_inverse(curr_loop)

    print(len(valid_closures))
    # Construct ground set

    ground_set = list(valid_closures)
    time_start = mytime.time()
    det_best_set, det_best_value = deterministicUSM(ground_set, oracle_function)
    time_deterministic = mytime.time()
    det_order_best_set, det_order_best_value = deterministicUSM_with_ordering(ground_set, oracle_function)
    time_deterministic_order = mytime.time()
    greedy_best_set, greedy_best_value = double_greedy(ground_set, oracle_function)
    time_greedy = mytime.time()
    greedy_order_best_set, greedy_order_best_value = double_greedy_with_ordering(ground_set, oracle_function)
    time_greedy_order = mytime.time()
    iterative_best_set, iterative_best_value = iterative_greedy(ground_set, oracle_function)
    time_iterative = mytime.time()
    print(f"det_best_value: {det_best_value} with {len(det_best_set)} edges, "\
          + f"det_order_best_value: {det_order_best_value} with {len(det_order_best_set)} edges, "
          + f"greedy_best_value: {greedy_best_value} with {len(greedy_best_set)} edges "\
          + f"greedy_order_best_value: {greedy_order_best_value} with {len(greedy_order_best_set)} edges "\
          + f"iter_best_value: {iterative_best_value} with {len(iterative_best_set)} edges.")
    print(f"initial_obj_value: {initial_obj_value}")

    J_det = det_best_value - initial_obj_value
    J_det_order = det_order_best_value - initial_obj_value
    J_greedy = greedy_best_value - initial_obj_value
    J_greedy_order = greedy_order_best_value - initial_obj_value
    J_iter = iterative_best_value - initial_obj_value
    print("\n Increased value: \n")
    print(f"J_det: {J_det},\n J_det_order: {J_det_order},\n J_greedy: {J_greedy},\n J_greedy_order: {J_greedy_order},\n J_iter: {J_iter}")

    max_value = max([J_det, J_det_order, J_greedy, J_greedy_order, J_iter])
    print(f"LB of half-optimal: {max_value/2}")


