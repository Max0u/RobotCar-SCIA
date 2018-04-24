import sys

sys.path.insert(0, './src/')

from simulator import Simulator
from colors import White
from layers.layers import Background, DrawLines, Perspective, Crop, Symmetric
from layers.noise import Shadows, Filter, NoiseLines, Enhance


simulator = Simulator()
white = White()

# draw lines on a background (top view)
#  -> nb background
#  -> nb rotations
#  -> nb crop
#  -> nb resize
simulator.add(Background(n_backgrounds=50, n_rot=5, n_crop=5, n_res=5, path='./ground_pics', input_size=(250, 200)))

# can adapt
#  -> thickness
#  -> color
#  -> radius
simulator.add(DrawLines(input_size=(250, 200), color_range=white))


#simulator.add(Perspective())
#simulator.add(Crop())
#simulator.add(Symmetric())
#simulator.add(Shadows())
#simulator.add(Filter(blur=0.5))
#simulator.add(Enhance(brightness=0.6))


simulator.generate(n_examples=100, path='my_dataset')
