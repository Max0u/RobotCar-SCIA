import numpy as np
import matplotlib.pyplot as plt

class Backtracking:

    size = 10
    def __init__(self):
        self.queue = np.array(self.size)
        self.index = 0

    def add_angle(self, angle):
        self.queue[self.index] = angle
        self.index = self.index + 1 % self.size

    def check_angle(self, angle):
        """
        if angle > pi / 2 (1.57) then back track on lasts angle
        """
        return True
