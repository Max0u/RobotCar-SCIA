import os

import threading
from random import randint


class GenerateThread (threading.Thread):
    def __init__(self, simulator, path, inter):
        threading.Thread.__init__(self)
        self.simulator = simulator
        self.path = path
        self.inter = inter

    def run(self):
        for i in range(self.inter[0], self.inter[1]):
            index = randint(0, len(self.simulator.input_images)-1)
            ii = self.simulator.input_images[index].copy()
            new_img, new_name, new_img2, new_name2= self.simulator.generate_one_image(ii)
            new_img.save(os.path.join(self.path, 'frame_' + str(i) + new_name))
            new_img2.save(os.path.join(self.path, 'frame_' + str(i) + new_name2))

