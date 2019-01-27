import numpy as np
from picamera import PiCamera
import time
import matplotlib.pyplot as plt
from picamera.array import PiRGBArray

from ironcar_light import Ironcar

iron = Ironcar()
iron.load_config()
iron.select_model("models/model-112,48-elu.h5")
iron.switch_mode("auto")

CAM_RESOLUTION = (320, 160)
loop = 20
true_fps = []
camera = iron.camera
start = 40
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
        prediction = iron.predict_from_img(img_arr)
        iron.mode_function(img_arr, prediction)
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

plt.savefig('best_fps.png')
m = max(true_fps)
true_fps = np.array(true_fps)

print("Best setting : " + str(range(start, end)[np.argmax(true_fps)]) + ", FPS : " + str(m))
