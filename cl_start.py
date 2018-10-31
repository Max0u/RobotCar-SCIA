from ironcar import *

model = "model-0,0YUV.h5"

iron = Ironcar()

iron.load_config()

iron.max_speed_update(0.5)

iron.select_model(model)


iron.switch_mode("auto")

#from threading import Thread
#iron.camera_thread = Thread(target=iron.camera_loop, args=())
#iron.camera_thread.start()

iron.camera_loop()
