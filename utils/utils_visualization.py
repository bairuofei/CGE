import time
import math
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_modified_path(graph: nx.graph, path: list = [], closures: list = [], savefig: bool = False):
    """ Plot path over graph. The closure edges are highlighted as red.
    """
    path_edge_set = set()
    if path:
        for i in range(len(path) - 1):
            path_edge_set.add((path[i], path[i+1]))

    fig, axes = plt.subplots(1, 2)
    for node in graph.nodes():
        x, y = graph.nodes()[node]["position"]
        axes[0].plot(x, y, 'o', markersize=2, color='g', alpha = 0.7)
        axes[1].plot(x, y, 'o', markersize=2, color='g', alpha = 0.7)
    for edge in graph.edges():
        node1, node2 = edge
        pose1, pose2 = graph.nodes()[node1]["position"], graph.nodes()[node2]["position"]
        if (node1, node2) in path_edge_set or (node2, node1) in path_edge_set:
            axes[0].plot([pose1[0], pose2[0]], [pose1[1], pose2[1]], '-', color='r', alpha = 0.5, zorder=5)
            axes[1].plot([pose1[0], pose2[0]], [pose1[1], pose2[1]], '-', color='r', alpha = 0.5, zorder=5)
        else:
            axes[0].plot([pose1[0], pose2[0]], [pose1[1], pose2[1]], '-', color='gray', alpha = 0.2)
            axes[1].plot([pose1[0], pose2[0]], [pose1[1], pose2[1]], '-', color='gray', alpha = 0.2)
    for node1, node2 in closures:
        pose1, pose2 = graph.nodes()[node1]["position"], graph.nodes()[node2]["position"]
        axes[1].plot([pose1[0], pose2[0]], [pose1[1], pose2[1]], '-', color='b', alpha = 0.7, zorder=5)
    axes[0].set_aspect('equal')
    axes[1].set_aspect('equal')
    if savefig:
        prefix = time.time()
        plt.savefig("./results/tsp_path" + str(math.floor(prefix)) + ".pdf")
    # plt.axis('equal')
    else:
        plt.show()


def plot_VRP_route(graph: nx.graph, num_robot: int, paths: list, candidate_loops: list = [], selected_loops: list = []):
    """ Plot vrp paths for multi-robots in different colors. """

    fig, axes = plt.subplots(1, num_robot+2)
    colors = ['b', 'g', 'c', 'm', 'y', 'r']

    # plot base graph for each subgraph
    for ax in axes:
        for node in graph.nodes():
            x, y = graph.nodes()[node]["position"]
            ax.plot(x, y, 'o', markersize=2, color='g', alpha = 0.2)
        for edge in graph.edges():
            node1, node2 = edge
            pose1, pose2 = graph.nodes()[node1]["position"], graph.nodes()[node2]["position"]
            ax.plot([pose1[0], pose2[0]], [pose1[1], pose2[1]], '-', color='gray', alpha = 0.1)
        ax.set_aspect('equal')
    
    # plot each robot's path separately; and plot all robots' paths at axes[0] and axes[N]
    for i, path in enumerate(paths):
        ax = axes[i+1]
        ax0 = axes[0]
        axN = axes[-1]
        path_position = [graph.nodes()[vertex]["position"] for vertex in path]
        ax.plot([position[0] for position in path_position], [position[1] for position in path_position],
                color=colors[i], alpha = 0.4)
        ax.scatter([position[0] for position in path_position], [position[1] for position in path_position],
                color=colors[i], alpha = 0.6)
        # Starting point marked as red
        ax.scatter([path_position[0][0]], path_position[0][1], color='r', marker = '*', s=80, alpha = 1, zorder=10)
        for special_ax in [ax0, axN]:
            alpha1 = 0.4
            alpha2 = 0.6
            alpha3 = 1
            markersize = 80
            if special_ax == ax0:
                alpha1 = 0.2
                alpha2 = 0.6
                alpha3 = 1
                markersize = 80
            special_ax.plot([position[0] for position in path_position], [position[1] for position in path_position],
                    color=colors[i], alpha = alpha1)
            special_ax.scatter([position[0] for position in path_position], [position[1] for position in path_position],
                    color=colors[i], alpha = alpha2)
            # Starting point marked as red
            special_ax.scatter([path_position[0][0]], path_position[0][1], marker = '*', s=80, color='r', alpha = alpha3, zorder=10)

    # plot candidate loop edges in axes[0] 
    ax = axes[0]
    for node_start, node_end in candidate_loops:
        pos_start = graph.nodes()[node_start]["position"]
        pos_end = graph.nodes()[node_end]["position"]
        ax.plot([pos_start[0], pos_end[0]], [pos_start[1], pos_end[1]], color='k', alpha = 1)
    # plot selected loop edges in axes[N]
    for node_start, node_end in selected_loops:
        pos_start = graph.nodes()[node_start]["position"]
        pos_end = graph.nodes()[node_end]["position"]
        # ax.plot([pos_start[0], pos_end[0]], [pos_start[1], pos_end[1]], color='r', alpha = 1)
        axN.plot([pos_start[0], pos_end[0]], [pos_start[1], pos_end[1]], color='r', alpha = 1)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

    return




def plot_multilayer(graph: nx.graph, num_robot: int, selected_loops: list = [], start_nodes: list = []):
    """ Plot 3D layered path for multiple robots. """

    colors = ['b', 'g', 'c', 'm', 'y', 'r']
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pos = {}
    for node in graph.nodes():
        x, y = graph.nodes()[node]["position"]
        z = int(node[-1])
        # z *= 2
        pos[node] = [x, y, z]


    ax.grid(False)

    # Plot nodes by each layer
    for node, position in pos.items():
        x, y, z = position
        if node not in start_nodes:
            color = colors[z]
            # color = 'b' if z == 1 else ('g' if z == 2 else 'c')
            ax.scatter(x, y, z, color=color, s=150)
            # ax.text(x, y, z, node, color='black')
        else:
            ax.scatter(x, y, z, color='r', marker="*", s=300)

    # Plot edges
    for edge in graph.edges():
        node1, node2 = edge
        x1, y1, z1 = pos[node1]
        x2, y2, z2 = pos[node2]
        if z1 == z2:
            color = colors[z1]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color=color)
        else:
            color = 'gray'
            ax.plot([x1, x2], [y1, y2], [z1, z2], '--', color=color)

    for edge in selected_loops:
        node1, node2 = edge
        x1, y1, z1 = pos[node1]
        x2, y2, z2 = pos[node2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], '--', color='red')
        # ax.plot([x1, x2], [y1, y2], [z1, z2], color='gray')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('white')
    ax.set_zticks([])

    plt.title('Three-layer Graph')
    plt.show()