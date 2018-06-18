import numpy as np
import matplotlib.image as mpimg
from keras.models import load_model
import cv2, os

import preprocess

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

#from ironcar import Ironcar

#ironcar = Ironcar()
file_path = "datasets/2018_06_13_16_44/frame_389_gas_0.0_dir_0.0.jpg"
img = mpimg.imread(file_path)
img = preprocess.preprocess(img)
img = np.array([img])

model_name = "./models/model-008.h5"
model = load_model(model_name)
print(model.predict(img))
