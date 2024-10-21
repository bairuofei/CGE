""" MILP solver for loop edge allocation."""

from ortools.linear_solver import pywraplp


class MILP_Solver():
    """
    1. set robot_dists, only contains the robot that involved in the loop edges
    2. set edge lens
    3. set edge_to_robot index. Key: edge_index, Value: (r1, r2)
    4. call solve
    """
    def __init__(self):
        return
    
    def set_robot_dists(self, robot_dists: list):
        self.robot_dists = robot_dists
        self.num_robot = len(self.robot_dists)
        return

    def set_edge_lens(self, edge_lens: list):
        self.edge_lens = edge_lens
        self.num_edge = len(self.edge_lens)
        return
    
    def set_edge_to_robot(self, edge_to_robot: dict):
        self.edge_to_robot = edge_to_robot
    
    def solve(self) -> list:
        """
        Variables: each edge has one boolean variable, and one max_value variable
        Constraints: (1) each robot has a max_value constraints
        Objective: minimize the max_value
        Results: a list of robot index that corresponds to loop edges
        """
        # Create linear program solver
        solver = pywraplp.Solver.CreateSolver('SCIP')

        # Create list of bool variables
        variables = [solver.BoolVar('x[%i]' % i) for i in range(self.num_edge)]

        # Create variable to represent max_value
        max_value = solver.BoolVar('max_value')
        max_value_upper_bound = sum(self.robot_dists) + sum(self.edge_lens)
        max_value.SetUb(max_value_upper_bound)

        # For each robot, create a list of coefficients for variables: [1, 0, -1]
        # 1 for x*dist，0 for 0，-1 for (1-x)*dist
        flag_lists = [[0 for _ in range(self.num_edge)] for _ in range(self.num_robot)]
        for i in range(self.num_edge):
            r1, r2 = self.edge_to_robot[i]
            flag_lists[r1][i] = 1
            flag_lists[r2][i] = -1

        for r in range(self.num_robot):
            solver.Add((self.robot_dists[r] + sum(variables[i]*self.edge_lens[i] if flag_lists[r][i] == 1 \
                                                  else (1-variables[i])*self.edge_lens[i] if flag_lists[r][i] == -1 \
                                                  else 0 for i in range(self.num_edge))) <= max_value)

        # Define objective function to minimize max_valu
        objective = solver.Minimize(max_value)

        # Solve the program
        solver.Solve()

        # Output results
        results = [-1 for _ in range(self.num_edge)]
        for i in range(self.num_edge):
            result = int(variables[i].solution_value())
            r1, r2 = self.edge_to_robot[i]
            if result == 1:
                results[i] = r1
            else:
                results[i] = r2
            
            # print('x[%i] =' % i, int(variables[i].solution_value()))
        print('Maximum distance after MILP allocation', int(max_value.solution_value()))

        return results


if __name__ == '__main__':

    robot_dists = [80, 100, 100]
    edge_lens = [12, 22, 13]
    edge_to_robot = {0: (0, 2), 1: (0, 1), 2: (1, 2)}

    milp_solver = MILP_Solver()
    milp_solver.set_robot_dists(robot_dists)
    milp_solver.set_edge_lens(edge_lens)
    milp_solver.set_edge_to_robot(edge_to_robot)

    results = milp_solver.solve()

    final_length = robot_dists[:]
    for i, edge_len in enumerate(edge_lens):
        final_length[results[i]] += edge_len
    print("Final path length:\n")
    print(final_length)
