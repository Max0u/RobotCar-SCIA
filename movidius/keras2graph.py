# Arda Mavi

import os
import sys
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json

def keras_to_tf(tf_model_path):
    saver = tf.train.Saver()
    with K.get_session() as sess:
        K.set_learning_phase(0)
        saver.save(sess, tf_model_path)
    return True

def tf_to_graph(tf_model_path, model_in, model_out, graph_path):
    os.system('mvNCCompile {0}.meta -in {1} -on {2} -o {3}'.format(tf_model_path, model_in, model_out, graph_path))
    return True

def keras_to_graph(model, model_in, model_out, graph_path, take_tf_files = False):
    # Getting Keras Model:
    keras_model = model

    # Saving TensorFlow Model from Keras Model:
    tf_model_path = './TF_Model/tf_model'
    keras_to_tf(tf_model_path)

    tf_to_graph(tf_model_path, model_in, model_out, graph_path)

    if take_tf_files == False:
        os.system('rm -rf ./TF_Model')
