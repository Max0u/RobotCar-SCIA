from ironcar_light import Ironcar

import signal
import sys

import argparse

def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

parser = argparse.ArgumentParser(description='Self-Driving Car Prediction Program')
parser.add_argument('-p', help='load model path', dest='model_path', type=str,\
        default="models/model-0,0YUV-squeeze48-ep100.h5")
parser.add_argument('-ms', help='max speed value', dest='max_speed', type=float,\
        default=0.4)
parser.add_argument('-ss', help='speed strategy', dest='speed_strat', type=str,\
        default="auto")
parser.add_argument('-v', help='verbose', dest='verb',
        type=s2b,   default='false')


args = parser.parse_args()

print('-' * 30)
print('Parameters')
print('-' * 30)
for key, value in vars(args).items():
    print('{:<20} := {}'.format(key, value))
print('-' * 30)


iron = Ironcar()

iron.load_config()

iron.verbose=args.verb

iron.max_speed_update(args.max_speed)

iron.select_model(args.model_path)

iron.switch_mode("auto")

iron.switch_speed_mode(args.speed_strat)

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
