import numpy as np
from picamera import PiCamera
import time
import matplotlib.pyplot as plt

CAM_RESOLUTION = (48, 48)
loop = 100
true_fps = []
camera = PiCamera()
start = 5
end = 90
for fps in range(start, end):
    print(fps)
    camera.framerate = fps
    camera.resolution = CAM_RESOLUTION
    save_time = time.time()
    for i in range(loop):
        output = np.empty((48, 48, 3), dtype=np.uint8)
        camera.capture(output, 'rgb', use_video_port=True)
        img_arr = output[:48, :48, :]
    true_fps.append(loop/(time.time()-save_time))
plt.plot(range(start, end), range(start, end), color='green')
plt.plot(range(start, end), true_fps)
plt.savefig('testcam.png')
