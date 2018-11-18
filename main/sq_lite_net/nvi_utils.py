import cv2, os
import numpy as np
import matplotlib.image as mpimg

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 48, 80, 1 #48, 112, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file, crop):
    """
    Load RGB images from a file
    """
    image = mpimg.imread(os.path.join(data_dir, image_file.strip()))
    if crop :
        image = image[60:-20,:,:]
    return image


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

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

def rgb2ycrcb(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    #img_ycrcb[:,:,0] = cv2.equalizeHist(img_ycrcb[:,:,0])
    return img_ycrcb

def bright_contr_auto(image):
    maxi = np.max(image)
    mini = np.min(image)
    
    alpha = 254/(maxi-mini)
    beta = -mini * alpha
    return image * 254/maxi

def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    #image = resize(image)
    #image = bright_contr_auto(image)
    image = rgb2yuv(image)
    #image = rgb2ycrcb(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = np.minimum(hls[:, :, 1][cond] * s_ratio, 255)
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + (np.random.rand() - 0.5) * 1.8
    hsv[:,:,2] = np.minimum(hsv[:,:,2], 255/ratio) * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(data_dir, center, steering_angle, range_x=50, range_y=10,
        crop=False):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = load_image(data_dir, center, crop), steering_angle
    image = resize(image)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size,
        is_training, crop):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(len(image_paths)):
            center = image_paths[index]
            steering_angle = steering_angles[index][1]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center,
                        steering_angle, crop=crop)
            else:
                image = load_image(data_dir, center, crop)
                image = resize(image)
            # add the image and steering angle to the batch
            images[i] = (preprocess(image)[:,:,0]).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
            steers[i] = steering_angle
            i += 1

            if i == batch_size:
                break
        yield images, steers
