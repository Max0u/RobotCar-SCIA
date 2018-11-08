from keras.models import load_model
import cv2
import preprocess
import time
import numpy as np
import md

from keras import backend as K

from keras.applications.mobilenet_v2 import MobileNetV2

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


mod = md.build_model_mobile()
#mod = load_model("model-test-lstm.hdf5", custom_objects={'root_mean_squared_error': root_mean_squared_error})

#mod = load_model("model-mobilenet-test.h5")
mod.load_weights("model-mobilenet-test.h5")

it = 1000
file_path = "test.jpg"
start_time = time.time()

for i in range(it):
    img = cv2.imread(file_path)
    img, _ = preprocess.preprocess(img)
    img = np.array([img])
    mod.predict(img)

print("--- %s fps ---" % str(it/(time.time() - start_time)))

