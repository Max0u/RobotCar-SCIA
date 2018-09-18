import cv2, os
import numpy as np
import matplotlib.image as mpimg
import scipy

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def brightness(image, factor):
    """
    Adjust brightness of an image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * factor
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def greyscale(image):
    """
    Convert an image to greyscale
    """
    greyscale = cv2.cvtColor(res5, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(res6, cv2.COLOR_GRAY2RGB)

def contrast(image, factor):
    """
    Contrast an image with a factor between 0 and 1
    """
    return np.array(255 * (image / 255)**factor, dtype=np.uint8)

def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def autobright(image, th):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        maxi = np.amax(hsv[:,:,2])
        factor = th / maxi
        hsv[:,:,2] = hsv[:,:,2] * factor
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def autobright_win(image, th, winsize):
    img = image.copy()
    for i in range(image.shape[0]-winsize):
        for j in range(image.shape[1]-winsize):
            img[i:i+winsize, j:j+winsize] = autobright(image[i:i+winsize, j:j+winsize], th)
    return image


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = autobright(image, 250)
    
    img = image.copy()

    image = scipy.misc.imresize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    #print(image.shape)
    #image = resize(image)
    #image = bright_contr_auto(image)
    #image = brightness(image, 3)
    #image = greyscale(image)
    #image = contrast(image, 0.65)

    image = rgb2yuv(image)
    #print(image.shape)
    #image = rgb2ycrcb(image)
    return image, img
