import cv2
import numpy as np
from math import acos
import seaborn as sns

def get_colors():
    colormap = sns.color_palette()
    element = [ colormap[0] ]
    element += colormap[2:5]
    element.append(colormap[6])
    element.append(colormap[9])
    colors = np.asarray(element) * (np.ones((len(element), 3)) * 255)

    return colors

class Obstacle:
    def __init__(self):
        self.colors = get_colors()

    def draw_rect(self, img):
        # position can be everywhere on the image except the last pixels to have space to put the object
        x = np.random.randint(img.shape[1] * 7 / 10)
        y = np.random.randint(img.shape[0] * 7 / 10)
        # object cannot be bigger than 30% of the image width
        x_size = np.random.randint(img.shape[1] * 3 / 10)
        # object cannot be bigger than 30% of the image height
        y_size = np.random.randint(img.shape[0] *  3 / 10)
        # color of the object
        color = self.colors[np.random.randint(len(colors))]
        # draw colored rectangle
        cv2.rectangle(img, (x, y), (x + x_size, y + y_size), color, cv2.FILLED)
        
        # compute direction vector to get angle to obstacle
        center_point = (x + x_size // 2, y + y_size // 2)
        car_pos = (img.shape[1] // 2, img.shape[0])
        
        vect = np.asarray(center_point) - np.asarray(car_pos)
        vect[1] = -vect[1]
        vect = vect / np.linalg.norm(vect, ord=2)
        angle = acos(vect[1])
        angle = angle if vect[0] > 0 else -angle
        
        return angle
