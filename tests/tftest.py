import tensorflow as tf

tf.load_op_library("tflite-lib/libtensorflow-lite.so")

tf.contrib.lite.Interpreter(model_path="models/model-0,0YUV-squeeze48.tflite")
