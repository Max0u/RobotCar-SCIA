from Adafruit_PCA9685 import PCA9685
import time


pwm = PCA9685()
pwm.set_pwm_freq(60)


def dir(value):
    """Sends the pwm signal on the dir channel"""
    if pwm is not None:
        pwm.set_pwm(2, 0, value)
        print('DIR : ', value)


def main():
    for i in range(1000):
        dir(i)
        time.sleep(0.1)
    return

main()
