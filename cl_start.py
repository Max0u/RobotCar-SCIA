from ironcar_light import *

import signal
import sys

import argparse

parser = argparse.ArgumentParser(description='Self-Driving Car Prediction Program')
parser.add_argument('-p', help='load model path', dest='model_path', type=str, default="models/model-0,0YUV.h5")

args = parser.parse_args()

print('-' * 30)
print('Parameters')
print('-' * 30)
for key, value in vars(args).items():
    print('{:<20} := {}'.format(key, value))
print('-' * 30)


iron = Ironcar()

iron.load_config()

iron.max_speed_update(0.5)

iron.select_model(args.model_path)

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
