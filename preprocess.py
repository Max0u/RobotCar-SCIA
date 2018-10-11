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

def autobrightness(img, dump=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    varG, meanG = var_mean(hsv)
    row, col, pix= img.shape
    nb_br, nb_bc, nb_bp = (3,5,1)
    box_row, box_col, box_pix = (row//nb_br,col//nb_bc,pix//nb_bp)
    boxs, boxs_hsv = [], []
    means, variances = [], []
    for i in range(nb_br):
        for j in range(nb_bc):
            boxs.append(img[i*box_row:(i+1)*box_row,j*box_col:(j+1)*box_col,:])
            boxs_hsv.append(hsv[i*box_row:(i+1)*box_row,j*box_col:(j+1)*box_col,:])
    for b in boxs_hsv:
        var, m = var_mean(b)
        means.append(m)
        variances.append(var)
    varM = np.mean(variances)
    for i in range(15):
        if dump:
            plt.subplot(3,5,i+1)
        if variances[i] >= varM:
            if dump:
                print("{}:{}".format(i+1,variances[i]))
            boxs_hsv[i][:,:,2] = np.floor(fn((boxs_hsv[i][:,:,2]/100)-0.5)).astype(np.uint8)
        else:
            boxs_hsv[i][:,:,2] = boxs_hsv[i][:,:,2]*0 + 10
            #faire en sorte de diminuer l'intensite de zone homogene
        boxs[i] = cv2.cvtColor(boxs_hsv[i],cv2.COLOR_HSV2RGB)
        variances[i], means[i] = var_mean(boxs_hsv[i])
        if dump:
            plt.imshow(boxs[i])
            plt.title("M={:.2}".format(means[i]))
    res = img.copy()
    for i in range(nb_br):
        for j in range(nb_bc):
            res[i*box_row:(i+1)*box_row,j*box_col:(j+1)*box_col,:] = boxs[i]
    return res



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
    image = image[60:-20, :, :]
    #image = image[80:, :, :]

    #if args.autob :
    #    image = autobright(image, 250)
    
    img = image.copy()

    image = scipy.misc.imresize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    #image = autobright(image, 250)
    #print(image.shape)
    #image = resize(image)
    #image = bright_contr_auto(image)
    #image = brightness(image, 3)
    #image = greyscale(image)
    #image = contrast(image, 0.65)
    #if args.yuv :
    image = rgb2yuv(image)
    #print(image.shape)
    #image = rgb2ycrcb(image)
    return image, img
