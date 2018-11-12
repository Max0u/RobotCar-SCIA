import numpy as np
from picamera import PiCamera
import time
import matplotlib.pyplot as plt

CAM_RESOLUTION = (200, 146)
loop = 100
true_fps = []

for fps in range(1,90):
    camera = PiCamera(framerate=fps)
    camera.resolution = CAM_RESOLUTION
    save_time = time.time()
    for i in range(loop):
        output = np.empty((160, 208, 3), dtype=np.uint8)
        camera.capture(output, 'rgb', use_video_port=True)
        img_arr = output[:146, :200, :]
    true_fps.append(loop/(time.time()-save_time))

plt.plot(range(1,90), true_fps)
plt.show()
