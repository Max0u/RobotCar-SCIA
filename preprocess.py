import cv2, os
import numpy as np
import matplotlib.image as mpimg
import scipy

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)



def var_mean(b):
    var = 0
    cont = 0
    m = np.mean(b[:,:,2])
    for i in b[:,:,2]:
        for j in i:
            cont+=1
            var += (m-j)**2
    return var/cont, m

def fn(x):
        return (100/(1+np.exp(-20*x)))

class prepro_args():
    def __init__(self):
        self.autob = False
        self.yuv = True
    def switch_yuv():
        self.yuv = not self.yuv
    def switch_autob():
        self.autob = not self.autob


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

def blur(img):
    size = 15

    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    # applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_motion_blur)
    return img, output


def preprocess(image):
    """
    Combine all preprocess functions into one
    """

    image = image[60:-20, :, :]
    #image = image[80:, :, :]
    
    #image = scipy.misc.imresize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    #image = autobright(image, 250)

    image = rgb2yuv(image)
    return image
