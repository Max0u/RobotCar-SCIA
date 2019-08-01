from keras.models import load_model
import cv2
import preprocess
import time
import numpy as np
import md
import cv2
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


mod = md.build_model_squeeze()
#mod = load_model("model-test-lstm.hdf5", custom_objects={'root_mean_squared_error': root_mean_squared_error})

#mod = load_model("model-mobilenet-test.h5")
mod.load_weights("model-sq48-112,48-ep90.h5")

it = 1000
file_path = "test.jpg"
start_time = time.time()

for i in range(it):
    img = cv2.imread(file_path)
    img = preprocess.preprocess(img)
    img = np.array([img])
    mod.predict(img)

print("--- %s fps ---" % str(it/(time.time() - start_time)))

