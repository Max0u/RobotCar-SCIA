import os

import threading
from random import randint


class GenerateThread (threading.Thread):
    def __init__(self, simulator, path, frame):
        threading.Thread.__init__(self)
        self.simulator = simulator
        self.path = path
        self.frame = frame

    def run(self):
        index = randint(0, len(self.simulator.input_images)-1)
        ii = self.simulator.input_images[index].copy()
        new_img, new_name, new_img2, new_name2= self.simulator.generate_one_image(ii)
        new_img.save(os.path.join(self.path, 'frame_' + str(self.frame) + new_name))
        new_img2.save(os.path.join(self.path, 'frame_' + str(self.frame) + new_name2))

