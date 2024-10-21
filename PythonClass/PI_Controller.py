import math


class PI_Controller():
    """ A PID controller with bounded output for target tracking. """
    def __init__(self, kp_d=1, kp_theta=1, ki_d=0.6, dist_tolerance=1, theta_tolerance=math.pi/18):
        self.kp_d = kp_d
        self.kp_theta = kp_theta
        self.ki_d = ki_d
        self.dist_tolerance = dist_tolerance    # distance to target tolerance
        self.theta_tolerance = theta_tolerance

        self.dist_integral = 0
        self.dist_integral_lim = 30  # upper limit of the integral term
        self.lin_vel_lim = 90        # upper limit of velocity

        self.reach_target = False

    def __compute_dist(self, current_pos, target_pos):
        dist = math.sqrt((current_pos[0]-target_pos[0]) ** 2 +
                         (current_pos[1]-target_pos[1]) ** 2)
        if dist <= self.dist_tolerance:
            dist = 0
            self.dist_integral = 0
            self.reach_target = True
        else:
            self.dist_integral += dist
            if self.dist_integral > self.dist_integral_lim:
                self.dist_integral = self.dist_integral_lim
        return dist

    def __compute_angle_diff(self, robot_direction, target_direction):
        angle_diff = target_direction-robot_direction
        if angle_diff < -1*math.pi:
            angle_diff += 2*math.pi
        elif angle_diff > math.pi:
            angle_diff -= 2*math.pi
        return angle_diff

    def control_input(self, robot):
        current_pos = [robot.x, robot.y]
        target_pos = [robot.target_x, robot.target_y]
        dist = self.__compute_dist(current_pos, target_pos)
        target_direction = math.atan2(target_pos[1]-current_pos[1],
                                      target_pos[0]-current_pos[0])
        angle_diff = self.__compute_angle_diff(robot.theta, target_direction)

        lin_vel = self.kp_d*dist+self.ki_d*self.dist_integral
        if lin_vel > self.lin_vel_lim:
            lin_vel = self.lin_vel_lim
        return lin_vel, self.kp_theta*angle_diff