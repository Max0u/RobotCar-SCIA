from keras.models import load_model
import cv2
import preprocess
import time
import numpy as np
import md

mod = md.build_model()
mod.load_weights("model-0,0YUV.h5")
it = 1000
file_path = "test.jpg"
start_time = time.time()

for i in range(it):
    img = cv2.imread(file_path)
    img, _ = preprocess.preprocess(img)
    img = np.array([img])
    mod.predict(img)

print("--- %s fps ---" % str(it/(time.time() - start_time)))

