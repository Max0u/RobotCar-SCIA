import os
import json

from PIL.Image import fromarray as PIL_convert
from utils import ConfigException, CameraException

import xbox

CONFIG = 'config.json'
CAM_RESOLUTION = (250, 150)
get_default_graph = None  # For lazy imports


class XboxCameraRecorder:
    capture_path = 'records'
    verbose = False
    max_speed_rate = 0.4

    def __init__(self):
        self.joy = xbox.Joystick()
        while not self.joy.connected():
            pass
        self.image_index = 0
        # PWM setup
        try:
            from Adafruit_PCA9685 import PCA9685

            self.pwm = PCA9685()
            self.pwm.set_pwm_freq(60)
        except Exception as e:
            print('The car will not be able to move')
            print('Are you executing this code on your laptop?')
            print('The adafruit error: ', e)
            self.pwm = None

        self.load_config()

        from threading import Thread

        self.camera_thread = Thread(target=self.camera_loop, args=())
        self.camera_thread.start()

    def gas(self, value):
        """Sends the pwm signal on the gas channel"""

        if self.pwm is not None:
            self.pwm.set_pwm(self.commands['gas_pin'], 0, value)
            if self.verbose:
                print('GAS : ', value)
        else:
            if self.verbose:
                print('GAS : ', value)

    def dir(self, value):
        """Sends the pwm signal on the dir channel"""

        if self.pwm is not None:
            self.pwm.set_pwm(self.commands['dir_pin'], 0, value)
            if self.verbose:
                print('DIR : ', value)
        else:
            if self.verbose:
                #print('PWM module not loaded')
                print('DIR : ', value)

    def camera_loop(self):

        from io import BytesIO
        from base64 import b64encode

        try:
            from picamera import PiCamera
            from picamera.array import PiRGBArray
        except Exception as e:
            print('picamera import error : ', e)

        try:
            cam = PiCamera(framerate=self.fps)
        except Exception as e:
            print('Exception ', e)
            raise CameraException()

        cam.resolution = CAM_RESOLUTION
        cam_output = PiRGBArray(cam, size=CAM_RESOLUTION)
        stream = cam.capture_continuous(cam_output, format="rgb", use_video_port=True)

        for f in stream:
            img_arr = f.array
            im = PIL_convert(img_arr)

            reverse = -self.joy.leftTrigger()
            gas = self.joy.rightTrigger() * self.max_speed_rate * (reverse if reverse != 0 else 1)
            if gas < 0:
                new_value = self.commands['stop']
            elif gas < 0.05:
                new_value = self.commands['neutral']
            else:
                new_value = int(
                    gas * (self.commands['drive_max']-self.commands['drive']) + self.commands['drive'])
            self.gas(new_value)

            direction = 1.3 * self.joy.leftX()
            if direction == 0:
                new_value = self.commands['straight']
            else:
                new_value = int(
                    direction * (self.commands['right'] - self.commands['left'])/2. + self.commands['straight'])
            self.dir(new_value)
            
            image_name  = ''.join([
                    'frame_',
                    str(self.image_index), 
                    '_gas_',
                    str(gas),
                    '_dir_', 
                    str(direction), 
                    '.jpg'
                ])
            image_name = os.path.join(self.capture_path, image_name)
            im.save(image_name)
            self.image_index += 1
            cam_output.truncate(0)

    def load_config(self):
        """Loads the config file of the ironcar
        Tests if all the necessary fields are present:
            - 'commands'
            - 'dir_pin'
            - 'gas_pin'
            - 'left'
            - 'straight'
            - 'right'
            - 'stop'
            - 'neutral'
            - 'drive'
            - 'drive_max'
            - invert_dir'
            - 'fps'
            - 'datasets_path'
            - 'stream_path'
            - 'models_path'
        """

        if not os.path.isfile(CONFIG):
            raise ConfigException('The config file `{}` does not exist'.format(CONFIG))

        with open(CONFIG) as json_file:
            config = json.load(json_file)

        # Verify that the config file has the good fields
        error_message = '{} is not present in the config file'
        for field in ['commands', 'fps', 'datasets_path', 'stream_path', 'models_path']:
            if field not in config:
                raise ConfigException(error_message.format(field))

        for field in ["dir_pin", "gas_pin", "left", "straight", "right", "stop",
                      "neutral", "drive", "drive_max", "invert_dir"]:
            if field not in config['commands']:
                raise ConfigException(error_message.format('[commands][{}]'.format(field)))

        self.commands = config['commands']

        self.fps = config['fps']

        # Folder to save the stream in training to create a dataset
        # Only used in training mode
        from datetime import datetime

        ct = datetime.now().strftime('%Y_%m_%d_%H_%M')
        self.save_folder = os.path.join(config['datasets_path'], str(ct))
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Folder used to save the stream when the stream is on
        self.stream_path = config['stream_path']
        if not os.path.exists(self.stream_path):
            os.makedirs(self.stream_path)

        return config
