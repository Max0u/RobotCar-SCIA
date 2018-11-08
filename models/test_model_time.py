from keras.models import load_model
import cv2
import preprocess
import time
import numpy as np
import md

from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


#mod = md.build_model()
#mod = load_model("model-test-lstm.hdf5", custom_objects={'root_mean_squared_error': root_mean_squared_error})

from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D 
from keras.layers import ReLU

with CustomObjectScope({'relu6': ReLU(6.),'DepthwiseConv2D': DepthwiseConv2D}):
    mod = load_model("model-mobilenet-test.h5")
#mod.load_weights("model-0,0YUV.h5")

it = 1000
file_path = "test.jpg"
start_time = time.time()

for i in range(it):
    img = cv2.imread(file_path)
    img, _ = preprocess.preprocess(img)
    img = np.array([img])
    mod.predict(img)

print("--- %s fps ---" % str(it/(time.time() - start_time)))

