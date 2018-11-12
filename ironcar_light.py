import os
import json
import numpy as np

from app import socketio
from PIL.Image import fromarray as PIL_convert
from utils import ConfigException, CameraException

import preprocess
import md

from collections import deque
from picamera import PiCamera

import time

CONFIG = 'config.json'
CAM_RESOLUTION = (200, 146)
get_default_graph = None  # For lazy imports

top, bot = 40, -40

class Ironcar():
    """Class of the car. Contains all the different fields, functions needed to
    control the car.
    """

    def __init__(self):

        self.mode = 'resting'  # resting, training, auto or dirauto
        self.speed_mode = 'constant'  # constant, confidence or auto
        self.started = False  # If True, car will move, if False car won't move.
        self.model = None
        self.current_model = None  # Name of the model
        self.graph = None
        self.curr_dir = 0
        self.curr_gas = 0
        self.max_speed_rate = 0.5
        self.model_loaded = False
        self.streaming_state = False

        self.n_img = 0
        self.save_number = 0
        self.load_config() 
        self.camera = PiCamera(framerate=self.fps)

        self.speed_acc = 0

        self.queue = deque(maxlen=50)

        self.verbose = False
        self.mode_function = self.default_call

        self.last_pred = time.time()
        self.count = 0

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


        #self.camera_loop()
        from threading import Thread

        self.camera_thread = Thread(target=self.camera_loop, args=())
        self.camera_thread.start()

    def camera_loop(self):
        """Makes the camera take pictures and save them.
        This loop is executed in a separate thread.
        """

        from io import BytesIO
        from base64 import b64encode

        try:
            from picamera import PiCamera
            from picamera.array import PiRGBArray
        except Exception as e:
            print('picamera import error : ', e)

        try:
            cam = self.camera
            
        except Exception as e:
            print('Exception ', e)
            raise CameraException()


        cam.resolution = CAM_RESOLUTION
        #cam_output = PiRGBArray(cam, size=CAM_RESOLUTION)
        #stream = cam.capture_continuous(cam_output, format="rgb", use_video_port=True)


        #for f in stream:
        while True:
            output = np.empty((160, 208, 3), dtype=np.uint8)
            cam.capture(output, 'rgb', use_video_port=True)

            if self.verbose and self.count == 1:
                print(cam.exposure_speed)
            img_arr = output[:146, :200, :]
            
            #if self.count == 8:
            #    import sys
            #    img = PIL_convert(img_arr, 'RGB')
            #    img.save('my.png')
            #    img.show()
            #    sys.exit(0)

            #cam_output.truncate(0)
            prediction = 0
            # Predict the direction only when needed
            if self.started:
                prediction = float(self.predict_from_img(img_arr))
            self.mode_function(img_arr, prediction)

            

    def gas(self, value):
        """Sends the pwm signal on the gas channel"""

        if self.pwm is not None:
            self.pwm.set_pwm(self.commands['gas_pin'], 0, value)
            #if self.verbose:
            #    print('GAS : ', value)
        else:
            if self.verbose:
                print('PWM module not loaded')
                #print('GAS : ', value)

    def dir(self, value):
        """Sends the pwm signal on the dir channel"""

        if self.pwm is not None:
            self.pwm.set_pwm(self.commands['dir_pin'], 0, value)
            #if self.verbose:
            #    print('DIR : ', value)
        else:
            if self.verbose:
                print('PWM module not loaded')
                #print('DIR : ', value)

    def default_call(self, img, prediction):
        """Default function call. Does nothing."""

        pass

    def kalman(self, preds):
        # fonction kalman
        # intial parameters
        n_iter = len(preds)
        sz = (n_iter,) # size of array
        x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
        z = preds # observations (normal about x, sigma=0.1)

        Q = 1e-5 # process variance

        # allocate space for arrays
        xhat=np.zeros(sz)      # a posteri estimate of x
        P=np.zeros(sz)         # a posteri error estimate
        xhatminus=np.zeros(sz) # a priori estimate of x
        Pminus=np.zeros(sz)    # a priori error estimate
        K=np.zeros(sz)         # gain or blending factor

        R = 0.01**2 # estimate of measurement variance, change to see effect

        # intial guesses
        xhat[0] = 0.0
        P[0] = 1.0

        for k in range(1,n_iter):
                # time update
                xhatminus[k] = xhat[k-1]
                Pminus[k] = P[k-1]+Q

                # measurement update
                K[k] = Pminus[k]/( Pminus[k]+R )
                xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
                P[k] = (1-K[k])*Pminus[k]
    
        return xhat[-1]

    def speed_strat(self, prediction):
        if abs(prediction) < 0.2 :
            speed_mode_coef =  1.5 + 0.2 * self.speed_acc 
            self.speed_acc += 1
            self.speed_acc = min(self.speed_acc, 5)
            prediction *= abs(prediction)
        else:
            if self.speed_acc > 3 :        
                speed_mode_coef = 0.1
                self.speed_acc -= 1
            else :
                speed_mode_coef = 1
        return prediction, speed_mode_coef



    def autopilot(self, img, prediction):
        """Sends the pwm gas and dir values according to the prediction of the
        Neural Network (NN).

        img: unused. But has to stay because other modes need it.
        prediction: dir val
        """
        """
        if abs(prediction) < 0.4 : 
            self.queue.append(prediction)
        else :
            self.queue.clear()
        """
        if self.started :

            speed_mode_coef = 1

            if self.speed_mode == 'confidence' :
                speed_mode_coef = 1.5 - min(prediction**2, .5)
            elif self.speed_mode == 'auto' :
                prediction, speed_mode_coef = self.speed_strat(prediction)

            # TODO add filter on direction to avoid having spikes in direction
            # TODO add filter on gas to avoid having spikes in speed
            #print('speed_mode_coef: {}'.format(speed_mode_coef))

            local_dir = prediction

            local_gas = self.max_speed_rate * speed_mode_coef

            if local_gas > 0:
                gas_value = int(local_gas * (self.commands['drive_max'] - self.commands['drive'])
                        + self.commands['drive'])
            else:
                gas_value = self.commands['stop']
                """
                gas_value = int(local_gas * (self.commands['rev_drive_max'] -
                    self.commands['rev_drive']) + self.commands['rev_drive']))
                """

            dir_value = int(local_dir * (self.commands['right'] - self.commands['left'])/2. + self.commands['straight'])
        else:
            gas_value = self.commands['neutral']
            dir_value = self.commands['straight']

        self.gas(gas_value)
        self.dir(dir_value)
        
        if self.streaming_state :
            self.training(img, prediction)

        if self.count == 10:
            now = time.time()
            if self.verbose :
                print("FPS : " + str(self.count/(now-self.last_pred)))
            self.last_pred = now
            self.count = 0
        self.count += 1

    def switch_mode(self, new_mode):
        """Switches the mode between:
                - training
                - resting
                - dirauto
                - auto
        """

        # always switch the starter to stopped when switching mode
        self.started = False

        # Stop the gas before switching mode and reset wheel angle (safe)
        self.gas(self.commands['neutral'])
        self.dir(self.commands['straight'])

        if new_mode == "auto":
            self.mode = 'auto'
            if self.model_loaded:
                self.mode_function = self.autopilot
            else:
                if self.verbose:
                    print("model not loaded")
        else:
            self.mode = 'resting'
            self.mode_function = self.default_call
            self.gas(self.commands['stop'])
            self.dir(self.commands['straight'])

        # Make sure we stopped and reset wheel angle even if the previous mode
        # sent a last command before switching.


        if self.verbose:
            print('switched to mode : ', new_mode)

    def on_start(self):
        """Switches started mode between True and False."""

        self.started = not self.started
        if self.verbose:
            print('starter set to {}'.format(self.started))
        return self.started

    def on_dir(self, data):
        """Triggered when a value from the keyboard/gamepad is received for dir.

        data: intensity of the key pressed.
        """

        if not self.started:
            return

        self.curr_dir = self.commands['invert_dir'] * float(data)
        if self.curr_dir == 0:
            new_value = self.commands['straight']
        else:
            new_value = int(
                self.curr_dir * (self.commands['right'] - self.commands['left'])/2. + self.commands['straight'])
        self.dir(new_value)

    def on_gas(self, data):
        """Triggered when a value from the keyboard/gamepad is received for gas.

        data: intensity of the key pressed.
        """

        if not self.started:
            return

        self.curr_gas = float(data) * self.max_speed_rate

        if self.curr_gas < 0:
            new_value = self.commands['stop']
        elif self.curr_gas == 0:
            new_value = self.commands['neutral']
        else:
            new_value = int(
                self.curr_gas * (self.commands['drive_max']-self.commands['drive']) + self.commands['drive'])
        self.gas(new_value)

    def max_speed_update(self, new_max_speed):
        """Changes the max_speed of the car."""

        self.max_speed_rate = new_max_speed
        if self.verbose:
            print('The new max_speed is : ', self.max_speed_rate)
        return self.max_speed_rate

    def predict_from_img(self, img):
        """Given the 250x150 image from the Pi Camera.

        Returns the direction predicted by the model (float)
        """
        try: 
            img = preprocess.preprocess(img)
            img = np.array([img])
            with self.graph.as_default():
                pred = float(self.model.predict(img, batch_size=1))
                if self.verbose:
                    print('pred : ', pred)
        
        except Exception as e:
            # Don't print if the model is not relevant given the mode
            if self.mode in ['dirauto', 'auto']: #self.verbose and self.mode in ['dirauto', 'auto']:
                print('Prediction error : ', e)
            pred = 0

        return pred

    def switch_speed_mode(self, speed_mode):
        """Changes the speed mode of the car"""
        self.speed_mode = speed_mode

    def switch_streaming(self):
        """Switches the streaming state."""

        self.streaming_state = not self.streaming_state
    
        camera = self.camera
        if self.streaming_state :
            camera.start_preview()
            camera.start_recording('videos/video.h264')
        else :
            camera.stop_recording()
            camera.stop_preview()
            
        if self.verbose:
            print('Streaming state set to {}'.format(self.streaming_state))


    def select_model(self, model_name):
        """Changes the model of autopilot selected and loads it."""

        if model_name == self.current_model:
            return

        try:
            # Only import tensorflow if needed (it's heavy)
            global get_default_graph
            if get_default_graph is None:
                try:
                    from tensorflow import get_default_graph
                    from keras.models import load_model
                except Exception as e:
                    if self.verbose:
                        print('ML error : ', e)
                    return

            if self.verbose:
                print('Selected model: ', model_name)
            
            self.model = md.build_model_squeeze();
            self.model.load_weights(model_name)

            #self.model = load_model(model_name)
            
            self.graph = get_default_graph()
            self.current_model = model_name

            self.model_loaded = True
            self.switch_mode(self.mode)

            if self.verbose:
                print('The model {} has been successfully loaded'.format(self.current_model))

        except Exception as e:
            if self.verbose:
                print('An Exception occured : ', e)

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
