from ironcar_light import *

import signal
import sys

model = "models/model-0,0YUV.h5"

iron = Ironcar()

iron.load_config()

iron.max_speed_update(0.5)

iron.select_model(model)

iron.switch_mode("auto")

iron.switch_speed_mode("constant")

def signal_handlerC(sig, frame):
    iron.switch_mode("resting")
    sys.exit(0)

def signal_handlerZ(sig, frame):
    iron.on_start()

signal.signal(signal.SIGINT, signal_handlerC)
signal.signal(signal.SIGTSTP, signal_handlerZ)
print('Press Ctrl+Z to start/stop')
print('Press Ctrl+C to quit')
signal.pause()
