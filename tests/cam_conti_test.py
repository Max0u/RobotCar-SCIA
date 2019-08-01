import numpy as np
from picamera import PiCamera
import time
import matplotlib.pyplot as plt
from picamera.array import PiRGBArray

CAM_RESOLUTION = (200, 146)
loop = 10
true_fps = []
camera = PiCamera()
start = 10
end = 90

cam_output = PiRGBArray(camera, size=CAM_RESOLUTION)
stream = camera.capture_continuous(cam_output, format="rgb", use_video_port=True)

for fps in range(start, end):
    print(fps)
    camera.framerate = fps
    camera.resolution = CAM_RESOLUTION
    save_time = time.time()
    count = 0
    for f in stream:
        img_arr = f.array
        cam_output.truncate(0)
        count += 1
        if count == loop:
            break
    true_fps.append(loop/(time.time()-save_time))


plt.plot(range(start, end), range(start, end), color='green', label='Theory')
plt.plot(range(start, end), true_fps, label='Practice')
plt.xlabel("FPS Set")
plt.ylabel("FPS Get")
plt.legend()

plt.savefig('testcamcontinu.png')
