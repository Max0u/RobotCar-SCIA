from ironcar_light import *

import signal
import sys

model = "model-0,0YUV.h5"

iron = Ironcar()

iron.load_config()

iron.max_speed_update(0.5)

iron.select_model(model)


iron.switch_mode("auto")

def signal_handler(sig, frame):
    iron.switch_mode("resting")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
print('Press Ctrl+C to quit')
signal.pause()

#from threading import Thread
#iron.camera_thread = Thread(target=iron.camera_loop, args=())
#iron.camera_thread.start()

iron.camera_loop()
