"""This file includes the implementation of VRP_solver, and test the basic function of submodular optimization.
Simply treate this file as an implementation of VRP_solver class.
"""

import networkx as nx

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from utils.utils_visualization import plot_VRP_route



class VRP_solver:
    """ Input: data = {distance matrix, num_vehicles, depot}, time_limit for solver in second
    Objective: minimize the length of the longest single route among all vehicles
    """
    def __init__(self, data, time_limit: float = 20):
        self.data = data
        # Create the routing index manager.
        self.manager = pywrapcp.RoutingIndexManager(
                len(data["distance_matrix"]), data["num_vehicles"], data["starts"], data["ends"]
            )
        # Create Routing Model.
        self.routing = pywrapcp.RoutingModel(self.manager)

        self.transit_callback_index = self.routing.RegisterTransitCallback(self.__distance_callback)
        # Define cost of each arc.
        # Can also set for each vehicle separately
        self.routing.SetArcCostEvaluatorOfAllVehicles(self.transit_callback_index)

        # Add Distance constraint.
        dimension_name = "Distance"

        # Dimension is an object used to keep track of each robot's cumulative distance
        # It is used to enforce the satisfaction of capacity constraints
        self.routing.AddDimension(
            self.transit_callback_index,
            0,  # no slack. Use this when the problem involves waiting, and this is the allowed maximum waiting time at one place
            50000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name,
        )
        self.distance_dimension = self.routing.GetDimensionOrDie(dimension_name)
        self.distance_dimension.SetGlobalSpanCostCoefficient(100)

        ## Setting first solution heuristic.
        self.search_parameters = pywrapcp.DefaultRoutingSearchParameters()

        # Method1: set search strategy: greedy cheapest search
        # self.search_parameters.first_solution_strategy = (
        #     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        # )

        # Method2: local search metaheruistics
        self.search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        self.search_parameters.time_limit.seconds = time_limit
        self.search_parameters.log_search = True

        return

    # Create and register a transit callback.
    # Note: here user defined distance callback can be used
    def __distance_callback(self, from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        # the routing variable index is internally used in ortools and can be ignored safely
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.data["distance_matrix"][from_node][to_node]

    def solve(self):
        self.solution = self.routing.SolveWithParameters(self.search_parameters)
        return
    
    def save_solution(self):
        """Save final path to self.final_path; save final distance to self.final_distance"""
        self.final_path = []
        self.final_distance = []
        for vehicle_id in range(self.data["num_vehicles"]):
            this_path = []
            this_distance = 0
            index = self.routing.Start(vehicle_id)
            this_path.append(self.manager.IndexToNode(index))
            while not self.routing.IsEnd(index):
                previous_index = index
                # How to find the next index for vehicle_id? Here vehicle_id is not specified.
                index = self.solution.Value(self.routing.NextVar(index))  # Get next index in final routing
                this_distance += self.routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
                this_path.append(self.manager.IndexToNode(index))
            self.final_path.append(this_path)
            self.final_distance.append(this_distance)
        return 
    
    def print_solution(self):
        """Prints solution on console."""
        print(f"Objective: {self.solution.ObjectiveValue()}")

        for vehicle_id in range(self.data["num_vehicles"]):
            plan_output = f"Route for vehicle {vehicle_id}:\n"
            route_distance = 0
            for node in self.final_path[vehicle_id]:
                plan_output += f" {node} -> "
            plan_output += "\n"
            plan_output += f"Distance of the route: {self.final_distance[vehicle_id]}m\n"
            print(plan_output)

        max_route_distance = max(self.final_distance)
        print(f"Maximum of the route distances: {max_route_distance}m")



    def get_final_path(self) -> list:
        return self.final_path
    
    def get_final_distance(self) -> list:
        return self.final_distance



def test_VRP():
    locations = [(456, 320), # location 0 - the depot
            (228, 0),    # location 1
            (912, 0),    # location 2
            (0, 80),     # location 3
            (114, 80),   # location 4
            (570, 160),  # location 5
            (798, 160),  # location 6
            (342, 240),  # location 7
            (684, 240),  # location 8
            (570, 400),  # location 9
            (912, 400),  # location 10
            (114, 480),  # location 11
            (228, 480),  # location 12
            (342, 560),  # location 13
            (684, 560),  # location 14
            (0, 640),    # location 15
            (798, 640)]  # location 16

    data = {}
    # Note: the distance marix must only contain integers
    data["distance_matrix"] = [
        # fmt: off
    [0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662],
    [548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210],
    [776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754],
    [696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358],
    [582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244],
    [274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708],
    [502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480],
    [194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662, 742, 856],
    [308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320, 1084, 514],
    [194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274, 810, 468],
    [536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730, 388, 1152, 354],
    [502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308, 650, 274, 844],
    [388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536, 388, 730],
    [354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342, 422, 536],
    [468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 0, 764, 194],
    [776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 0, 798],
    [662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 0],
        # fmt: on
    ]
    data["num_vehicles"] = 4
    data["depot"] = 0   # the index of the depot in distance matrix


    graph = nx.Graph()
    for i, position in enumerate(locations):
        graph.add_node(i, position = position)

    vrp_solver = VRP_solver(data)
    vrp_solver.solve()
    vrp_solver.save_solution()
    vrp_solver.print_solution()

    # Draw
    plot_VRP_route(graph, 4, vrp_solver.get_final_path())

         
    
if __name__ == "__main__":
    pass