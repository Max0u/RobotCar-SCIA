import numpy as np
import math
import matplotlib.pyplot as plt

class Cartography:
    # in meter
    cur_pos = np.asarray([0, 0])
    # in meter per second
    cur_speed = np.asarray([0, 0])

    # list of successive positions
    path = []
    # in sec
    dt = 1
    # in meter
    road_width = 0.75
    # in meter
    epsilon = 0.01
    # vectors
    displacements = []
    circuit = []

    def __init__(self, road_width = 0.75, epsilon=0.01):
        self.road_width = road_width
        self.epsilon = epsilon
        self.path.append(self.cur_pos)
        self.circuit.append(self.cur_pos)

    # angle in unit rad
    def get_distance(self, angle):
        shift = (angle / math.pi) + (math.pi / 2)
        cos = math.cos(shift)
        # value below varies depending on time taken to be on center
        dist = np.linalg.norm(cos * self.cur_speed, ord=2)
        print(dist)
        return dist

    def add_position(self, acceleration, angle):
        # has made a complete turn
        if len(self.path) > 20 and \
           np.linalg.norm(self.cur_pos, ord=2) < self.epsilon:
                return

        self.cur_speed = acceleration * self.dt + self.cur_speed
        self.cur_pos = self.cur_speed * self.dt + self.cur_pos
        displacement = self.cur_pos - self.path[-1]
        self.displacements.append(displacement)

        distance_from_center = np.zeros(2)
        if np.any(displacement):
            norm_v = np.asarray([ -displacement[1], displacement[0] ])
            normal_vect = norm_v / np.linalg.norm(norm_v, ord=2)
            distance_from_center = self.get_distance(angle) * normal_vect
            if np.cross(normal_vect, displacement) < 0:
                distance_from_center = -distance_from_center
        self.path.append(self.cur_pos + distance_from_center)
        self.circuit.append(self.cur_pos)

    def draw_circuit(self):
        self.path = np.asarray(self.path)
        self.displacements = np.asarray(self.displacements)
        self.circuit = np.asarray(self.circuit)
        x = self.path[:, 0]
        y = self.path[:, 1]

        fig, ax = plt.subplots()
        ax.scatter(x, y, alpha=1, c='green')

        x_p = self.circuit[:, 0]
        y_p = self.circuit[:, 1]

        plt.plot(x, y)
        plt.plot(x_p, y_p, 'grey', alpha = 0.6)

        """
        # computing left and right points
        left = []
        right = []
        normals = [ np.asarray([-v[1], v[0]]) for v in self.displacements ]
        normals = [ n / np.linalg.norm(n, ord=1) * self.road_width for n in \
                normals ]

        disp = self.displacements
        #disp = np.zeros(self.displacements.shape)
        for i in range(len(normals)):
            if np.cross(self.displacements[i], normals[i]) > 0:
                left.append(self.path[i] + (disp[i] / 2) + normals[i])
                right.append(self.path[i] + (disp[i] / 2) - normals[i])
            else:
                left.append(self.path[i] + (disp[i] / 2) - normals[i])
                right.append(self.path[i] + (disp[i] / 2) + normals[i])

        left.append(left[0])
        right.append(right[0])
        left = np.asarray(left)
        right = np.asarray(right)
        plt.plot(left[:, 0], left[:, 1])
        plt.plot(right[:, 0], right[:, 1])
        """

        for i in range(0, len(x) - 1):
            # drawing red arrows
            size = (x[i + 1] - x[i]) / 2 + (y[i + 1] - y[i]) / 2
            if size != 0:
                plt.arrow(x[i], y[i], self.displacements[i, 0] / 2,
                        self.displacements[i, 1] / 2,
                        head_width=.1, head_length=0.2, color='red')
        plt.show()
