import cv2, os
import numpy as np
import matplotlib.image as mpimg
import scipy

from md import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """ 
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                

def preprocess(image):
    """
    Combine all preprocess functions into one
    """

    image = scipy.misc.imresize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    image = rgb2yuv(image)

    return image
