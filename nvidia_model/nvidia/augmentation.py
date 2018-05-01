# Visualizations will be shown in the notebook.
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2


# Setting up paths of our csv
path = '../datasets/behavioural_driving/Dataset_2'
standard_path = '../datasets/behavioural_driving/Dataset_2'
recovery_path = '../datasets/behavioural_driving/Recovery_Driving'
udacity_path = "../datasets/behavioural_driving/Udacity/data"
track2_path = '../datasets/behavioural_driving/Track_2'
img_path = path + '/IMG'
models_path = "./models"



########## Image Loading Function ##########

def read_img(img_full_path, img_dir="/IMG"):
    prefix_path = udacity_path + img_dir

    if "Dataset_2" in img_full_path:
        prefix_path = standard_path + img_dir
    elif "Recovery_Driving" in img_full_path:
        prefix_path = recovery_path + img_dir
    elif "Track_2" in img_full_path:
        prefix_path = track2_path + img_dir
    
    img_path = "{0}/{1}".format(prefix_path, img_full_path.split("/")[-1])    
    img = cv2.imread(img_path)
    
    # OpenCV reads images in BGR format, we are simply converting and returning the image in RGB format
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

########## Image Manipulation Functions ##########

def fliph_image(img):
    """
    Returns a horizontally flipped image
    """
    return cv2.flip(img, 1)


def blur_image(img, f_size=5):
    """
    Applies Gaussir Blur to smoothen the image.
    This in effect performs anti-aliasing on the provided image
    """
    img = cv2.GaussianBlur(img,(f_size, f_size),0)
    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)


# Read more about it here: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
def translate_image(img, st_angle, low_x_range, high_x_range, low_y_range, high_y_range, delta_st_angle_per_px):
    """
    Shifts the image right, left, up or down. 
    When performing a lateral shift, a delta proportional to the pixel shifts is added to the current steering angle 
    """
    rows, cols = (img.shape[0], img.shape[1])
    translation_x = np.random.randint(low_x_range, high_x_range) 
    translation_y = np.random.randint(low_y_range, high_y_range) 
    
    st_angle += translation_x * delta_st_angle_per_px

    translation_matrix = np.float32([[1, 0, translation_x],[0, 1, translation_y]])
    img = cv2.warpAffine(img, translation_matrix, (cols, rows))
    
    return img, st_angle

def change_image_brightness_rgb(img, s_low=0.2, s_high=0.75):
    """
    Changes the image brightness by multiplying all RGB values by the same scalacar in [s_low, s_high).
    Returns the brightness adjusted image in RGB format.
    """
    img = img.astype(np.float32)
    s = np.random.uniform(s_low, s_high)
    img[:,:,:] *= s
    np.clip(img, 0, 255)
    return  img.astype(np.uint8)


def add_random_shadow(img, w_low=0.6, w_high=0.85):
    """
    Overlays supplied image with a random shadow poligon
    The weight range (i.e. darkness) of the shadow can be configured via the interval [w_low, w_high)
    """
    cols, rows = (img.shape[0], img.shape[1])
    
    top_y = np.random.random_sample() * rows
    bottom_y = np.random.random_sample() * rows
    bottom_y_right = bottom_y + np.random.random_sample() * (rows - bottom_y)
    top_y_right = top_y + np.random.random_sample() * (rows - top_y)
    if np.random.random_sample() <= 0.5:
        bottom_y_right = bottom_y - np.random.random_sample() * (bottom_y)
        top_y_right = top_y - np.random.random_sample() * (top_y)

    
    poly = np.asarray([[ [top_y,0], [bottom_y, cols], [bottom_y_right, cols], [top_y_right,0]]], dtype=np.int32)
        
    mask_weight = np.random.uniform(w_low, w_high)
    origin_weight = 1 - mask_weight
    
    mask = np.copy(img).astype(np.int32)
    cv2.fillPoly(mask, poly, (0, 0, 0))
    #masked_image = cv2.bitwise_and(img, mask)
    
    return cv2.addWeighted(img.astype(np.int32), origin_weight, mask, mask_weight, 0).astype(np.uint8)


########## Data Augmentation Function ##########

def augment_image(img, st_angle, p=1.0):
    """
    Augment a given image, by applying a series of transformations, with a probability p.
    The steering angle may also be modified.
    Returns the tuple (augmented_image, new_steering_angle)
    """
    aug_img = img
    
    #if np.random.random_sample() <= 1.0:
        # Reduce aliasing via blurring
        #aug_img = blur_image(aug_img)
   
    if np.random.random_sample() <= p: 
        # Horizontally flip image
        aug_img = fliph_image(aug_img)
        st_angle = -st_angle
     
    if np.random.random_sample() <= p:
        aug_img = change_image_brightness_rgb(aug_img)
    
    if np.random.random_sample() <= p: 
        aug_img = add_random_shadow(aug_img, w_low=0.45)
            
    if np.random.random_sample() <= p:
        # Shift the image left/right, up/down and modify the steering angle accordingly
        aug_img, st_angle = translate_image(aug_img, st_angle, -60, 61, -20, 21, 0.35/100.0)
    
    # TODO In the future try adding slight rotations
        
    return aug_img, st_angle


########## Image Generator Function ##########

def generate_images(df, target_dimensions, img_types, st_column, st_angle_calibrations, batch_size=100, shuffle=True, 
                    data_aug_pct=0.8, aug_likelihood=0.5, st_angle_threshold=0.05, neutral_drop_pct=0.25):
    """
    Generates images whose paths and steering angle are stored in supplied dataframe object df
    Returns the tuple (batch,steering_angles)
    """
    # e.g. 160x320x3 for target_dimensions
    batch = np.zeros((batch_size, target_dimensions[0],  target_dimensions[1],  target_dimensions[2]), dtype=np.float32)
    steering_angles = np.zeros(batch_size)
    df_len = len(df)
    
    while True:
        k = 0
        while k < batch_size:            
            idx = np.random.randint(0, df_len)       

            for img_t, st_calib in zip(img_types, st_angle_calibrations):
                if k >= batch_size:
                    break
                                
                row = df.iloc[idx]
                st_angle = row[st_column]            
                
                # Drop neutral-ish steering angle images with some probability
                if abs(st_angle) < st_angle_threshold and np.random.random_sample() <= neutral_drop_pct :
                    continue
                    
                st_angle += st_calib                                                                
                img_type_path = row[img_t]  
                img = read_img(img_type_path)                
                
                # Resize image
                    
                img, st_angle = augment_image(img, st_angle, p=aug_likelihood) if np.random.random_sample() <= data_aug_pct else (img, st_angle)
                batch[k] = img
                steering_angles[k] = st_angle
                k += 1
            
        yield batch, np.clip(steering_angles, -1, 1)            

