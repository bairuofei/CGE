import time
import copy
import math
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
import matplotlib.animation as animation

import networkx as nx

from utils.utils_general import save_data, read_data

from PythonClass.PI_Controller import PI_Controller


class Quadcoptor:
    """ A robot that follows a task_list, with a PID controller for trajectory generation. """
    def __init__(self, init_pose: list = [0, 0, 0], task_list: list = [], controller_para: list = [2, 15], \
                 body_color='k', trace_color='b', name='', text_name=''):
        self.x = init_pose[0]
        self.y = init_pose[1]
        self.theta = init_pose[2]      # radian
        self.radius = 2
        self.body_color = body_color
        self.trace_color = trace_color
        self.special_trace_color = 'r'  # trace that approaching vertex that has index in indicator
        self.task_list = task_list
        self.task_index = 1   # Next target index, not current target index
        self.controller = PI_Controller(
            kp_d=controller_para[0], kp_theta=controller_para[1])
        self.indicator = set()

        self.target_x = self.x
        self.target_y = self.y

        self.name = name
        self.text_name = text_name

        Path = mpath.Path
        self.drone_outline = [
            (Path.MOVETO, [0, 0]),
            (Path.LINETO, [1, 0.7]),
            (Path.LINETO, [2.5, -0.3]),
            (Path.LINETO, [3, 0.2]),
            (Path.LINETO, [1, 1.7]),
            (Path.LINETO, [0, 2.7]),
            (Path.LINETO, [-1, 1.7]),
            (Path.LINETO, [-3, 0.2]),
            (Path.LINETO, [-2.5, -0.3]),
            (Path.LINETO, [-1, 0.7]),
            (Path.CLOSEPOLY, [0, 0])]
        self.model_scale = 2  # 2.5

        self.hover_lin_vel = 20
        self.hover_ang_vel = 6

        self.trace_lines = []

    def __set_theta(self, raw_theta):
        self.theta = raw_theta
        if self.theta < -1*math.pi:
            self.theta += 2*math.pi
        elif self.theta > math.pi:
            self.theta -= 2*math.pi
    
    def __get_current_pos(self):
        return self.x, self.y, self.theta

    def __set_target(self, task_to_pose: dict):
        """ Set next target position for robot """
        region_name = self.task_list[self.task_index]
        self.target_x, self.target_y = task_to_pose[region_name]
        self.task_index += 1

    def __modify_motor_collection(self):
        """ Re-draw the robot in its current position. """
        codes, verts = zip(*self.drone_outline)
        path = mpath.Path(np.array(verts)*self.model_scale, codes)
        path = path.transformed(mpl.transforms.Affine2D(
        ).rotate_deg(-90+math.degrees(self.theta)))
        path.vertices += np.array([self.x, self.y])
        drone_path_patch = mpatches.PathPatch(path)
        drone_patches = []
        drone_patches.append(drone_path_patch)
        motor_collection = PatchCollection(
            drone_patches, match_original=False, facecolor=self.body_color, alpha=0.8)
        return motor_collection
   
    def set_indicator(self, indicators: set):
        """ The set of indices that are special along this robot's path. Painted in different color"""
        self.indicator = copy.deepcopy(indicators)
        return

    def update_robot_pose(self, task_to_pose: dict):
        if self.controller.reach_target:  
            # self.modify_robot_state(is_task_complete)
            if self.task_index == len(self.task_list):
                lin_vel = self.hover_lin_vel
                ang_vel = self.hover_ang_vel
                print(self.name + " finished.")
            else:
                self.__set_target(task_to_pose)
                self.controller.reach_target = False
                return
        else:
            [lin_vel, ang_vel] = self.controller.control_input(self)
        self.last_x = self.x
        self.last_y = self.y
        self.x += dt*lin_vel*math.cos(self.theta)
        self.y += dt*lin_vel*math.sin(self.theta)
        raw_theta = self.theta + dt*ang_vel
        self.__set_theta(raw_theta)


    def paint_robot_new_trace(self, ax):
        trace_line_x = [self.last_x, self.x]
        trace_line_y = [self.last_y, self.y]
        linewidth = 3  # 1.5
        if self.task_index - 1 in self.indicator: # or self.task_index - 2 in self.indicator:
            color = self.special_trace_color
            alpha = 0.8
        else:
            color = self.trace_color
            alpha = 0.25
            if self.task_index - 2 in self.indicator:
                alpha = 0.08
        line = mlines.Line2D(trace_line_x, trace_line_y,
                             lw=linewidth, alpha=alpha, color=color)
        ax.add_line(line)

    def repaint_robot(self, ax):
        ax.add_collection(self.__modify_motor_collection())
        # ax.add_collection(self.modify_arm_collection())

    def repaint_robot_label(self, ax):
        ax.text(x=self.x, y=self.y+2*self.radius, s=self.text_name, fontsize=20)

    def paint_start_position(self, ax, task_to_pose: dict):
        target = self.task_list[0]
        pose = task_to_pose[target]
        ax.scatter(pose[0], pose[1], marker='*', s = 400, color='r', zorder = 10, alpha = 0.3)


class Environment():
    def __init__(self, task_to_pose: dict, range: list =[(0, 100), (0, 100)], graph: nx.Graph = None, graph_scale: float = 1):
        # region_list includes multiple lists, each list contains [shape，center coordinates，variable list, background color]
        self.x_range = range[0]
        self.y_range = range[1]
        self.region_list = task_to_pose
        self.graph = graph
        self.scale = graph_scale

    def paint_environment(self, ax):
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)

        markersize = 8
        linewidth = 5

        for node in self.graph.nodes():
            x, y = self.graph.nodes()[node]["position"]
            x *= self.scale
            y *= self.scale
            ax.plot(x, y, 'o', markersize=markersize, color='gray', alpha = 0.2)
        for edge in self.graph.edges():
            node1, node2 = edge
            pose1, pose2 = list(self.graph.nodes()[node1]["position"]), list(self.graph.nodes()[node2]["position"])
            for i in range(len(pose1)):
                pose1[i] *= self.scale
                pose2[i] *= self.scale
            ax.plot([pose1[0], pose2[0]], [pose1[1], pose2[1]], '-', color='gray', alpha = 0.1, linewidth = linewidth)
        
        print(len(ax.lines), len(ax.texts), len(ax.collections))
        return





def init():
    """ Init function for animation. Use outside reference to ax.
    Note this function will be called twice initially.
    """
    del ax.collections[:]
    del ax.lines[:]
    del ax.texts[:]
    battle_environment.paint_environment(ax)
    for robot in robot_list:
        robot.paint_start_position(ax, task_to_pose)

    num_plot_elements[0] = int(len(ax.lines))
    num_plot_elements[1] = int(len(ax.texts))
    num_plot_elements[2] = int(len(ax.collections))
    for robot in robot_list:
        robot.repaint_robot(ax)
    print(f"current: {len(ax.collections)}")
    return ax.collections


def animate(i):
    """ Counter function for animation. Use outside reference to ax. """
    print(i)
    print(num_plot_elements)
    if i > 10:
        del ax.collections[num_plot_elements[2]:]
        del ax.texts[num_plot_elements[1]:]
        if len(ax.texts) > len(task_to_pose):
            del ax.texts[len(task_to_pose):]
        # del ax.lines[num_plot_elements[0]: num_plot_elements[0]+10]

        for robot in robot_list:
            robot.update_robot_pose(task_to_pose)
            robot.repaint_robot(ax)
            robot.paint_robot_new_trace(ax)
            robot.repaint_robot_label(ax)
            
    return ax.collections+ax.lines+ax.texts


if __name__ == '__main__':
    dt = 0.05

    use_original = False  # whether add selected loop edges into the robot's routes when visualizing.
    name_mp4_file = "simulation_paper.mp4"
    save_video = False

    with open('config.yaml', 'r') as file:
        yaml_data = yaml.safe_load(file)  
    # Save path for video simulation
    path_save_graph = yaml_data["path_save_graph"]
    path_save_robot_path = yaml_data["path_save_robot_path"]
    path_save_original_robot_path = yaml_data["path_save_original_robot_path"]
    path_save_indicator = yaml_data["path_save_indicator"]

    ## Input: (1) task_to_pose, (2) robot task list

    # 1. Read graph environment
    env_graph = read_data(path_save_graph)
    graph_scale = 20

    # 2. Read robot paths
    final_paths = read_data(path_save_robot_path)
    num_robots = len(final_paths)

    # 3. Read special vertex indicator. Each robot has a corresponding set of indices
    all_indicator = read_data(path_save_indicator)  # list of sets

    # 4. Read robot original vrp paths for comparison
    vrp_paths = read_data(path_save_original_robot_path)


    if use_original:
        all_indicator = [set() for _ in range(num_robots)]
        robot_paths = vrp_paths
    else:
        robot_paths = final_paths


    task_to_pose = {}
    for one_path in robot_paths:
        for vertex in one_path:
            pose = list(env_graph.nodes()[vertex]["position"])
            for i in range(len(pose)):
                pose[i] *= graph_scale
            task_to_pose[vertex] = pose
    

    battle_environment = Environment(task_to_pose, range=[(-50, 350), (-50, 350)], graph = env_graph, graph_scale=graph_scale)

    robot_list = []
    colors = ['b', 'g', 'c', 'm', 'y', 'r', 'orange', 'brown']
    for r in range(num_robots):
        init_vertex = robot_paths[r][0]
        robot = Quadcoptor(init_pose=task_to_pose[init_vertex] + [0],\
                           task_list=robot_paths[r],\
                           body_color=colors[r], trace_color=colors[r], \
                           name="robot" +f"{r}", text_name="robot" +f"{r}")
        robot.set_indicator(all_indicator[r])
        robot_list.append(robot)

    fig, ax = plt.subplots(figsize=(15, 15))
    # ax.set_xlim(battle_environment.x_range)
    # ax.set_ylim(battle_environment.y_range)
    ax.axis('equal')
    ax.axis('off')

    # num_init_lines, num_init_texts, num_init_collections
    num_plot_elements = [0, 0, 0]

    # save_count is the number of frames to save
    ani = animation.FuncAnimation(
        fig, animate, init_func=init, interval=80, blit=True, save_count=2200)
    # ani.save('single_pendulum_nodecay.gif', writer='imagemagick')  # , fps=100
    if save_video:
        ani.save(name_mp4_file, fps=45, extra_args=['-vcodec', 'libx264'])
    plt.show()