import cv2
import os

image_folder = '../datasets/data'
video_name = 'video.avi'

def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

def frame_sort(x, y):
    return float(x.split("_")[1]) - float(y.split("_")[1])

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")], key=cmp_to_key(frame_sort))
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, 30, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
    print(image)
cv2.destroyAllWindows()
video.release()
