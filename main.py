"""This file is the final implementation of the whole algorithm, including how to allocate the selected loop edges to robots by MILP, 
and how to insert these edges into robot's routes."""

import numpy as np
import networkx as nx
import random
import time
import copy
import yaml
from collections import defaultdict

from utils.utils_visualization import plot_VRP_route, plot_multilayer
from TopoGraph import TopoGraph
from Oracle import Oracle
from PythonClass.MILP_solver import MILP_Solver

from submodular_maximization import double_greedy, double_greedy_with_ordering, deterministicUSM, \
    deterministicUSM_with_ordering, iterative_greedy, iterative_greedy_heap, double_greedy_with_ordering_heap, deterministicUSM_with_ordering_heap

from utils.utils_graph import get_d_opt, copy_node_with_new_name, write_nx_graph_to_g2o
from utils.utils_general import read_data, save_data

from VRP_solver import VRP_solver


def insert_loop_edge(loop_edges: list, vrp_paths: list, loop_allocation: list):
    """ Insert selected loop edges to robots' VRP paths. 
    Input:
        1. loop_edges, each vertex in a loop edge should have robot index suffix
        2. vrp_paths, each vertex in vrp_paths should have robot index suffix
        3. loop_allocation[i], the allocated robot index of the i-th loop edge
    Returns:
        1. final exploration paths for robots;
        2. a indicator set for each robot to record the index of loop-closing vertex, which will be used for video simulation
    """
    new_path= copy.deepcopy(vrp_paths)

    robot_to_allocated = defaultdict(set)
    for i, r in enumerate(loop_allocation):
        robot_to_allocated[r].add(i)
    
    # Determine loop position in vrp_path. 
    # If only one vertex in a loop edge in vrp path, then find it, and insert;
    # If two vertices are all in vrp paths, then loop-closing starts from the later vertex to earlier vertex
    all_indicator = []  # list of sets
    for r in range(len(vrp_paths)):  # for robot r
        robot_to_insert = {}
        for e_idx in robot_to_allocated[r]:
            this, that = loop_edges[e_idx]
            this_idx, that_idx = -1, -1
            if this in new_path[r]:
                this_idx = new_path[r].index(this)
            if that in new_path[r]:
                that_idx = new_path[r].index(that)
            if this_idx == -1:
                robot_to_insert[that_idx] = this
            elif that_idx == -1:
                robot_to_insert[this_idx] = that
            elif this_idx > that_idx: # two vertex all in robot's vrp path
                robot_to_insert[this_idx] = that
            else:
                robot_to_insert[that_idx] = this
        
        # Insert now
        insert_indices = list(robot_to_insert.keys())
        insert_indices.sort()
        offset = 0
        indicator_loop_closing = set()
        for idx_insert in insert_indices:
            starting_vertex = new_path[r][idx_insert + offset]
            loop_closing_vertex = robot_to_insert[idx_insert]
            new_path[r].insert(idx_insert+offset+1, loop_closing_vertex)
            indicator_loop_closing.add(idx_insert+offset+1)
            if idx_insert+offset+1 < len(new_path[r]) - 1:
                new_path[r].insert(idx_insert+offset+2, starting_vertex)
            offset += 2
        
        all_indicator.append(indicator_loop_closing)
    return new_path, all_indicator


def remove_path_node_suffix(paths: list) -> list:
    """ Remove robot-id suffix in the node name. """
    new_paths = []
    for r, path in enumerate(paths):
        new_paths.append([])
        for node in path:
            if node[-2] != '_':
                new_paths[-1].append(node)
            else:
                new_paths[-1].append(node[:-2])
    return new_paths



def test_insert():
    loop_edges = [(7, 6), (1, 9), (2, 8)]
    vrp_paths = [[1, 4, 7, 9], [2, 4, 6, 8, 10]]
    loop_allocation = [0, 0, 1]
    new_path, all_indicator = insert_loop_edge(loop_edges, vrp_paths, loop_allocation)
    print("New path:\n")
    print(new_path)
    print("Indicator:\n")
    print(all_indicator)


    
if __name__ == "__main__":

    with open('config.yaml', 'r') as file:
        yaml_data = yaml.safe_load(file)  
    topo_metric_graph_path = yaml_data["topo_metric_graph_path"]
    log_path = yaml_data["log_path"]
    g2o_graph_save_folder = yaml_data["g2o_graph_save_folder"]

    # Save path for video simulation
    path_save_graph = yaml_data["path_save_graph"]
    path_save_robot_path = yaml_data["path_save_robot_path"]
    path_save_original_robot_path = yaml_data["path_save_original_robot_path"]
    path_save_indicator = yaml_data["path_save_indicator"]

    random.seed(101)
    for curr_node_num in [10]:
        record_obj = []
        record_time = []
        for iter in range(1):
            ############################### Global parameters  ###################################
            alpha = 0.0135   # trade-off between distance and graph Laplacian. This value is not important
            para_lambda = 0.32  # Defines the final alpha
            num_vehicles = 3  
            node_num = curr_node_num  # Number of size for grid graph

            # The algorithms used, to save computational time
            use_dUSM = False

            save_time = round(time.time())

            # Default covariance matrix
            Cov = np.zeros((3, 3))
            Cov[0, 0] = 0.1
            Cov[1, 1] = 0.1
            Cov[2, 2] = 0.001
            Sigma = np.linalg.inv(Cov)


            # Save graph for repeated experiments
            save_graph = True  # whether save current random graph
            use_previous_graph = False  # whether use graph from graph_path
            previous_save_time = 1711003875
            if use_previous_graph:
                # graph_path = f"/home/ruofei/code/cpp/submodular_maximization/graph_save/prior_graph_r{num_vehicles}_{node_num}_{previous_save_time}.pickle"
                graph_path = topo_metric_graph_path + f"prior_graph_{node_num}_{previous_save_time}.pickle"
                save_time = previous_save_time
            else:
                graph_path = topo_metric_graph_path + f"prior_graph_r{num_vehicles}_{node_num}_{save_time}.pickle"


            if use_previous_graph:
                prior_graph = read_data(graph_path)
            else:
                ratio_remove_node = 0.1
                ratio_remove_edge = 0.3 * ratio_remove_node
                add_random_in_position = True
                prior_graph = TopoGraph()
                prior_graph.grid_2d_graph(node_num, node_num)
                prior_graph.add_node_position(add_random=add_random_in_position)
                prior_graph.map_name_to_str()
                prior_graph.add_edge_distance()
                prior_graph.random_remove_nodes(round(node_num*node_num*ratio_remove_node))
                prior_graph.random_remove_edges(round(node_num*node_num*ratio_remove_edge))

                ## Add covariance for edges
                prior_graph.add_edge_information_matrix(Sigma)
                prior_graph.add_edge_weight()

            if save_graph:
                save_data(prior_graph, graph_path)

            # Get distance matrix for VRP solver
            node_list = prior_graph.get_node_list()
            distance_matrix = prior_graph.get_distance_matrix(node_list, scale_to_int=True)


            ################################# 0. VRP solver ######################################
            # (0.1) Specify starting index
            start_position = (8, 8)
            start_idx = -1
            curr_distance = float("inf")
            for n_idx, node in enumerate(node_list):
                node_pose = prior_graph.graph.nodes()[node]["position"]
                if (start_position[0] - node_pose[0])**2 + (start_position[1] - node_pose[1])**2 < curr_distance:
                    curr_distance = (start_position[0] - node_pose[0])**2 + (start_position[1] - node_pose[1])**2
                    start_idx = n_idx

            # (0.2) Add dummy end location
            data = {}
            distance_matrix[:, start_idx] = 0
            data["distance_matrix"] = distance_matrix
            data["num_vehicles"] = num_vehicles

            # data["depot"] = start_idx, the index of the depot in distance matrix
            data["starts"] = [start_idx for _ in range(num_vehicles)]
            data["ends"] = [start_idx for _ in range(num_vehicles)]
            
            # (0.3) Solve VRP
            vrp_solver = VRP_solver(data, time_limit=60)
            vrp_solver.solve()
            vrp_solver.save_solution()
            vrp_solver.print_solution()
            vrp_path_distance_all = vrp_solver.get_final_distance()

            final_path_idx = vrp_solver.get_final_path()
            vrp_paths = []
            for k in range(num_vehicles):
                this_path = []
                for idx in final_path_idx[k]:
                    this_path.append(node_list[idx])
                this_path.pop()
                vrp_paths.append(this_path)

            # (0.4) Recover full path over prior graph
            for k in range(num_vehicles):
                vrp_paths[k] = prior_graph.connect_tsp_path(vrp_paths[k])
            

            ################## 1. Simulate multi-robot pose graph from VRP path ##################

            nodes_to_robots = defaultdict(set)  # This dict is reused in the following
            for k in range(num_vehicles):
                for node in vrp_paths[k]:
                    nodes_to_robots[node].add(k)

            # (1.1) Create combined base graph. 
            #       (a) copy base graph; (b) duplicate overlapped nodes; (c) Connect duplicate nodes.
            # To connect duplicate nodes, a chain connection is enough, because base_graph only used to compute edge length.
            base_graph = prior_graph.graph.copy()
            edge_attrs = {"distance": 0, "information": Sigma, "weight": get_d_opt(Sigma)}
            for node in nodes_to_robots.keys():
                if len(nodes_to_robots[node]) > 1:  # Repeated visited nodes
                    robots_here = list(nodes_to_robots[node])
                    robots_here.sort()
                    mapping = {node: node + f"_{robots_here[0]}"}
                    base_graph = nx.relabel_nodes(base_graph, mapping, copy=False)
                    for j in range(1, len(robots_here)):
                        base_graph = copy_node_with_new_name(base_graph, node + f"_{robots_here[0]}", node+f"_{robots_here[j]}", edge_attrs=edge_attrs)
                    for j in range(1, len(robots_here)):
                        base_graph.add_edge(node + f"_{robots_here[0]}", node+f"_{robots_here[j]}", **edge_attrs)


            # (1.2) Create combined pose graph. 
            #       (a) Create individual pose graph, rename repeated vertices
            #       (b) Rename repeated vertices in robots' VRP path
            #       (c) Connect individual graph as one collaborative pose graph
            G_vrp_individuals = []
            for k in range(num_vehicles):
                repeated_nodes = []
                vrp_edges = set()
                for j in range(len(vrp_paths[k])):
                    node = vrp_paths[k][j]
                    if len(nodes_to_robots[node]) > 1:
                        repeated_nodes.append(node)
                    if j < len(vrp_paths[k]) - 1:
                        vrp_edges.add((vrp_paths[k][j], vrp_paths[k][j+1]))
                        vrp_edges.add((vrp_paths[k][j+1], vrp_paths[k][j]))
                
                G_vrp_single = prior_graph.graph.edge_subgraph(vrp_edges).copy()

                # Rename repeated nodes that are visited by many robots
                for node in repeated_nodes:
                    mapping = {node: node+f"_{k}"}
                    G_vrp_single = nx.relabel_nodes(G_vrp_single, mapping, copy=False)
                G_vrp_individuals.append(G_vrp_single)
            
            # Rename repeated vertex in robots' VRP paths
            for k in range(num_vehicles):
                for i in range(len(vrp_paths[k])):
                    if len(nodes_to_robots[vrp_paths[k][i]]) > 1:
                        vrp_paths[k][i] = vrp_paths[k][i] + f"_{k}"
            
            # Connect individual graph as a combined pose graph
            robot_to_graph = [i for i in range(num_vehicles)]
            for node in nodes_to_robots.keys():
                if len(nodes_to_robots[node]) > 1:
                    robots_here = list(nodes_to_robots[node])
                    for i in range(len(robots_here)):
                        for j in range(i):
                            # connect robots_here[i] and robots_here[j]
                            g_idx1 = robot_to_graph[robots_here[i]]
                            g_idx2 = robot_to_graph[robots_here[j]]
                            node1 = node + f"_{robots_here[i]}"
                            node2 = node + f"_{robots_here[j]}"
                            curr_graph = G_vrp_individuals[g_idx2]
                            # add edge with parameters: (1) distance; (2) information; (3) weight
                            curr_graph.add_edge(node1, node2, distance=0, information=Sigma, weight=get_d_opt(Sigma))

                            if g_idx1 != g_idx2:
                                G_vrp_individuals[g_idx2] = nx.compose(curr_graph, G_vrp_individuals[g_idx1])
                                robot_to_graph[robots_here[i]] = g_idx2
            # Initially, the robots start at the same place, therefore, the pose graph must be connected
            # double check
            for i in range(num_vehicles):
                if robot_to_graph[i] != robot_to_graph[0]:
                    print("ERROR!!!")
            print("Invididual graphs composed as one graph.")
            G_vrp = G_vrp_individuals[robot_to_graph[0]]
            G_vrp_node_list = list(G_vrp.nodes())
            G_vrp_edge_list = set(G_vrp.edges())

                            
            ############################ 2. Create candidate loop edges ############################
            oracle_function = Oracle(alpha=alpha)
            oracle_function.set_base_graph(base_graph) 
            oracle_function.set_initial_graph(G_vrp, G_vrp_node_list)  

            valid_closures = set()
            checked_closures = set()
            
            # test_alpha = (initial_obj_value - oracle_function.get_dist_offset()) / sum(vrp_path_distance_all)
            time_start = time.time()
            count_checked = 0
            # FIXME: The candidate loop edge cannot identify repeated vertices of robots with different suffix
            max_ratio = 0
            min_ratio = float("inf")
            for i, curr in enumerate(G_vrp_node_list):
                if i % 50 == 0:
                    print(f"Create candidate loop edges, {i}/{len(G_vrp_node_list)}")
                for closure in G_vrp_node_list[:i]:
                    if (curr, closure) in G_vrp.edges() or (closure, curr) in G_vrp.edges() or curr == closure:
                        continue
                    if (curr, closure) in valid_closures or (closure, curr) in valid_closures:
                        continue  
                    if (curr, closure) in checked_closures or (closure, curr) in checked_closures:
                        continue
                    if curr[-1] != ')' and closure[-1] != ')' and curr[:-2] == closure[:-2]:   # Avoid loop closure at the same vertex
                        continue
                    count_checked += 1  
                    curr_loop = (closure, curr)
                    curr_ratio, curr_laplacian, curr_distance = oracle_function.ratio_combined([curr_loop])
                    if curr_ratio > 1000:
                        print(curr_loop)
                        print(f"{curr_laplacian}, {curr_distance}")
                    min_ratio = min(min_ratio, curr_ratio)
                    max_ratio = max(max_ratio, curr_ratio)
                    checked_closures.add(curr_loop)
                    valid_closures.add(curr_loop)

            # TODO: different lambda value
            new_alpha = min_ratio + (max_ratio - min_ratio) * para_lambda
            oracle_function.set_alpha(new_alpha)  

            initial_obj_value = oracle_function.gain_combined([])
            valid_closures_filtered = set()
            for loop_edge in valid_closures:
                curr_obj_value = oracle_function.gain_combined([loop_edge])
                if curr_obj_value > initial_obj_value:
                    valid_closures_filtered.add(loop_edge)

            print(f"We get ground set of size {len(valid_closures_filtered)} out of {count_checked} candidates.")
            # Construct ground set
            ground_set = list(valid_closures_filtered)

            time_create_candidate_edges = time.time() - time_start
            print(f"Time to create candidate loop edges: {time_create_candidate_edges} s")

            # plot_VRP_route(base_graph, num_vehicles, final_path_node, candidate_loops=list(ground_set))
                

            ########################### 3. Call submodular optimization ###########################
            time_start = time.time()
            if use_dUSM:
                det_best_set, det_best_value = deterministicUSM(ground_set, oracle_function)
            else:
                det_best_set = set()
                det_best_value = 0
            time_deterministic = time.time()
            if use_dUSM:
                det_order_best_set, det_order_best_value = deterministicUSM_with_ordering(ground_set, oracle_function)
            else:
                det_order_best_set = set()
                det_order_best_value = 0
            time_deterministic_order = time.time()
            greedy_best_set, greedy_best_value = double_greedy(ground_set, oracle_function)
            time_greedy = time.time()
            greedy_order_best_set, greedy_order_best_value = double_greedy_with_ordering(ground_set, oracle_function)
            time_greedy_order = time.time()
            iterative_best_set, iterative_best_value = iterative_greedy(ground_set, oracle_function)
            time_iterative = time.time()
            iterative_heap_best_set, iterative_heap_best_value = iterative_greedy_heap(ground_set, oracle_function)
            time_iterative_heap = time.time()
            greedy_order_heap_best_set, greedy_order_heap_best_value = double_greedy_with_ordering_heap(ground_set, oracle_function)
            time_greedy_order_heap = time.time()
            if use_dUSM:
                det_order_heap_best_set, det_order_heap_best_value = deterministicUSM_with_ordering_heap(ground_set, oracle_function)
            else:
                det_order_heap_best_set = set()
                det_order_heap_best_value = 0
            time_deterministic_order_heap = time.time()

            print(f"USM time: \n  det: {time_deterministic - time_start} s\n" +\
                f"  det_order: {time_deterministic_order - time_deterministic} s\n" +\
                f"  det_order_heap: {time_deterministic_order_heap - time_greedy_order_heap} s\n" +\
                f"  greedy: {time_greedy - time_deterministic_order} s\n" +\
                f"  greedy_order: {time_greedy_order - time_greedy} s\n" +\
                f"  greedy_order_heap: {time_greedy_order_heap - time_iterative_heap} s\n" +\
                f"  iterative_greedy: {time_iterative - time_greedy_order} s\n" +\
                f"  iterative_greedy_heap: {time_iterative_heap - time_iterative} s\n")

            print(f"det_best_value: {det_best_value} with {len(det_best_set)}/{len(ground_set)} edges, \n"\
                + f"det_order_best_value: {det_order_best_value} with {len(det_order_best_set)}/{len(ground_set)} edges, \n"
                + f"det_order_heap_best_value: {det_order_heap_best_value} with {len(det_order_heap_best_set)}/{len(ground_set)} edges, \n"
                + f"greedy_best_value: {greedy_best_value} with {len(greedy_best_set)}/{len(ground_set)} edges, \n"\
                + f"greedy_order_best_value: {greedy_order_best_value} with {len(greedy_order_best_set)}/{len(ground_set)} edges, \n"\
                + f"iter_best_value: {iterative_best_value} with {len(iterative_best_set)}/{len(ground_set)} edges, \n" \
                + f"iter_best_heap_value: {iterative_heap_best_value} with {len(iterative_heap_best_set)}/{len(ground_set)} edges.")
            print(f"initial_obj_value: {initial_obj_value}")

            J_det = det_best_value - initial_obj_value
            J_det_order = det_order_best_value - initial_obj_value
            J_greedy = greedy_best_value - initial_obj_value
            J_greedy_order = greedy_order_best_value - initial_obj_value
            J_iter = iterative_best_value - initial_obj_value

            print(f"Obj Increased value: \n  J_det: {J_det},\n  J_det_order: {J_det_order},\n" +\
                f"  J_greedy: {J_greedy},\n  J_greedy_order: {J_greedy_order},\n  J_iter: {J_iter}")
            max_value = max([J_det, J_det_order, J_greedy, J_greedy_order, J_iter])
            print(f"LB of half-optimal: {max_value/2}")
            print(f"\nactual_alpha: {new_alpha}, min_ratio: {min_ratio}, max_ratio: {max_ratio}")

            # Save log of this experiment
            with open(log_path, 'a') as file:
                file.write("----------------------------------\n")
                file.write(f"\nRobot num: {num_vehicles}, Iter: {iter} Graph size: {node_num} * {node_num}, Time: {save_time}, alpha = {alpha}\n")
                file.write(f"\nactual_alpha: {new_alpha}, min_ratio: {min_ratio}, max_ratio: {max_ratio}\n")
                file.write(f"Time to create candidate loop edges: {time_create_candidate_edges} s.\n")
                file.write(f"ground set of size {len(valid_closures_filtered)} out of {count_checked} candidates.\n")
                file.write(f"USM time: \n  det: {time_deterministic - time_start} s\n" +\
                            f"  det_order: {time_deterministic_order - time_deterministic} s\n" +\
                            f"  greedy: {time_greedy - time_deterministic_order} s\n" +\
                            f"  greedy_order: {time_greedy_order - time_greedy} s\n" +\
                            f"  iterative_greedy: {time_iterative - time_greedy_order} s\n")
                # Obj
                file.write(f"Obj value: \n det_best_value: {det_best_value} with {len(det_best_set)}/{len(ground_set)} edges, \n"\
                        + f" det_order_best_value: {det_order_best_value} with {len(det_order_best_set)}/{len(ground_set)} edges, \n"
                        + f" greedy_best_value: {greedy_best_value} with {len(greedy_best_set)}/{len(ground_set)} edges, \n"\
                        + f" greedy_order_best_value: {greedy_order_best_value} with {len(greedy_order_best_set)}/{len(ground_set)} edges, \n"\
                        + f" iter_best_value: {iterative_best_value} with {len(iterative_best_set)}/{len(ground_set)} edges.\n")
                # Increase
                file.write(f"Obj Increased value: \n J_det: {J_det},\n J_det_order: {J_det_order},\n" +\
                            f" J_greedy: {J_greedy},\n J_greedy_order: {J_greedy_order},\n J_iter: {J_iter}\n")
                file.write(f"LB of half-optimal: {max_value/2}\n\n")

            # Record log for multiple experiments comparison
            record_obj.append([J_det, J_det_order, J_greedy, J_greedy_order, J_iter])
            record_time.append([time_deterministic - time_start, \
                                time_deterministic_order - time_deterministic,\
                                time_deterministic_order_heap - time_greedy_order_heap,\
                                time_greedy - time_deterministic_order,\
                                time_greedy_order - time_greedy,\
                                time_greedy_order_heap - time_iterative_heap,\
                                time_iterative - time_greedy_order,\
                                time_iterative_heap - time_iterative])


            ############################## 4. GTSAM evaluation #####################################
            # (1) Save graph as g2o file
            # (2) GTSAM comparison
            # Record robots starting vertices
            node_robot_start = node_list[start_idx]
            nodes_start = []
            for i in range(num_vehicles):
                nodes_start.append(node_robot_start + f"_{i}")    

            need_pose_graph_evaluation = True
            if need_pose_graph_evaluation:
                save_path = g2o_graph_save_folder
                file_name_no_loop = f"{num_vehicles}robot_{node_num}_{node_num}_noLoop_{save_time}.g2o"
                G_vrp_for_g2o = copy.deepcopy(G_vrp)
                write_nx_graph_to_g2o(G_vrp, save_path + file_name_no_loop, nodes_start)

                file_name_plus_loop = f"{num_vehicles}robot_{node_num}_{node_num}_plusLoop_{save_time}.g2o"
                selected = iterative_best_set
                # Add selected edges into pose graph
                for node1, node2 in selected:  
                    # Distance does not matter here
                    G_vrp_for_g2o.add_edge(node1, node2, distance=0, information=Sigma, weight=get_d_opt(Sigma))
                write_nx_graph_to_g2o(G_vrp_for_g2o, save_path + file_name_plus_loop, nodes_start)
            

            ##################### 5. Visualize layered multi-robot pose graph ######################
            # (1) copy G_vrp, which has already added selected loop edges
            G_vrp_plot = copy.deepcopy(G_vrp)
            loop_edge_plot = list(iterative_best_set)
            start_nodes = copy.deepcopy(nodes_start)

            # (2) rename nodes in G_vrp
            G_vrp_plot_nodes = list(G_vrp_plot.nodes())
            for node in G_vrp_plot_nodes:
                if node[-2] != '_':
                    robot_id = list(nodes_to_robots[node])[0]
                    mapping = {node: node+f"_{robot_id}"}
                    G_vrp_plot = nx.relabel_nodes(G_vrp_plot, mapping, copy=False)
            
            # (3) rename nodes in selected loop edge
            for edge_idx in range(len(loop_edge_plot)):
                this, that = loop_edge_plot[edge_idx]
                if this[-2] != '_':
                    robot_id = list(nodes_to_robots[this])[0]
                    this += f"_{robot_id}"
                if that[-2] != '_':
                    robot_id = list(nodes_to_robots[that])[0]
                    that += f"_{robot_id}"
                loop_edge_plot[edge_idx] = (this, that)

            # (4) TODO: rename nodes in ground_set, for visulization
            ground_set_no_suffix = []
            for this, that in ground_set:
                if this[-2] != '_':
                    robot_id = list(nodes_to_robots[this])[0]
                    this += f"_{robot_id}"
                if that[-2] != '_':
                    robot_id = list(nodes_to_robots[that])[0]
                    that += f"_{robot_id}"
                ground_set_no_suffix.append((this, that))


            # (4) Plot G_vrp and loop edges
            plot_multilayer(G_vrp_plot, num_robot=num_vehicles, selected_loops=loop_edge_plot, start_nodes=start_nodes)
            
            # Draw
            plot_VRP_route(base_graph, num_vehicles, vrp_paths, \
                        candidate_loops=list(ground_set), selected_loops=list(iterative_best_set))
            
            ############################# 6. MILP loop edge allocation #############################
            # Reuse edges in loop_edge_plot
            # 1. set robot_dists
            # 2. set edge lens
            # 3. set edge_to_robot index
            use_MILP_allocate = True
            if use_MILP_allocate:
                print("Start MILP loop edge allocation......")
                robot_vrp_dists = vrp_path_distance_all[:]
                loop_edge_lens = []
                edge_to_robot = {}

                envolved_robot = set()
                for e_idx in range(len(loop_edge_plot)):
                    this, that = loop_edge_plot[e_idx]
                    r1, r2 = int(this[-1]), int(that[-1])
                    envolved_robot.add(r1)
                    envolved_robot.add(r2)
                    edge_to_robot[e_idx] = (r1, r2)
                    this_original = this[:-2]
                    that_original = that[:-2]
                    edge_length = prior_graph.get_path_length([this_original, that_original])
                    edge_length *= 100 * 2   # 100 for VRP int rounding, 2 for forth and back
                    loop_edge_lens.append(round(edge_length, 1))

                envolved_robot_list = list(envolved_robot)
                envolved_robot_list.sort()
                robot_vrp_dists = []
                for r in envolved_robot_list:
                    robot_vrp_dists.append(vrp_path_distance_all[r])

                # Call MILP solver
                milp_solver = MILP_Solver()
                milp_solver.set_robot_dists(robot_vrp_dists)
                milp_solver.set_edge_lens(loop_edge_lens)
                milp_solver.set_edge_to_robot(edge_to_robot)

                results = milp_solver.solve()

                final_length = vrp_path_distance_all[:]
                loop_allocation = []  # used for loop edge insertion
                for e_idx, edge_len in enumerate(loop_edge_lens):
                    assigned_robot_idx = results[e_idx]
                    actual_robot_idx = envolved_robot_list[assigned_robot_idx]
                    loop_allocation.append(actual_robot_idx) 
                    final_length[actual_robot_idx] += edge_len
                print("VRP path length:")
                print(vrp_path_distance_all)
                print("Final path length:")
                print(final_length)
                print("Finish MILP loop edge allocation.")


            ################################# 7. Loop edge insertion ################################
            # all loop edge in loop_edge_plot has robot index, but vrp_paths only has robot index for repeated vertex
            
            # Add robot index suffix to vrp_paths
            print("Inserting loop edges to robots......")
            copyed_vrp_paths = []
            for r, path in enumerate(vrp_paths):
                copyed_vrp_paths.append([])
                for node in path:
                    if node[-2] != '_':
                        copyed_vrp_paths[-1].append(node + f"_{r}")
                    else:
                        copyed_vrp_paths[-1].append(node)
            new_paths, all_indicator = insert_loop_edge(loop_edge_plot, copyed_vrp_paths, loop_allocation)
            print("Finish loop edge insertion.")


            #################### 8. Save graph and path for video simulation #######################
            use_simulation = True
            if use_simulation:
                # (1) save grpah environment to path_save_graph
                save_data(prior_graph.graph, path_save_graph)

                # (2) save robot path to path_save_robot_path
                # recover vrp_path to node list in prior_graph.graph
                robot_path_to_save = []
                for path in new_paths:
                    robot_path_to_save.append([])
                    for p in range(len(path)):
                        if path[p][-2] != "_":
                            robot_path_to_save[-1].append(path[p])
                        else:
                            robot_path_to_save[-1].append(path[p][:-2])
                save_data(robot_path_to_save, path_save_robot_path)

                # (3) save indicator of loop-closing vertex to path_save_indicator
                save_data(all_indicator, path_save_indicator)

                # (4) save original path of robots
                original_vrp_paths = remove_path_node_suffix(vrp_paths)
                save_data(original_vrp_paths, path_save_original_robot_path)


                
        num_methods_obj = 5
        num_methods_time = 8
        with open(log_path, 'a') as file:
            file.write("************************************\n")
            obj_records = ["" for _ in range(num_methods_obj)]
            for one_record in record_obj:
                for k in range(num_methods_obj):
                    obj_records[k] += f"{one_record[k]}, "
            time_records = ["" for _ in range(num_methods_time)]
            for one_record in record_time:
                for k in range(num_methods_time):
                    time_records[k] += f"{one_record[k]}, "
            file.write("Obj record: \n")
            for record_str in obj_records:
                file.write(record_str + "\n")
            file.write("Time record: \n")
            for record_str in time_records:
                file.write(record_str + "\n")
            
            
