import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, Activation, concatenate

import argparse
import os

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def fire(x, squeeze=16, expand=64):
    x = Conv2D(squeeze, (1,1), activation='elu', padding='valid')(x)
     
    left = Conv2D(expand, (1,1), activation='elu', padding='valid')(x)
     
    right = Conv2D(expand, (3,3), activation='elu', padding='same')(x)
   
    x = concatenate([left, right], axis=3)
    return x


def build_model():
    """
    Modified NVIDIA model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def build_model_squeeze():
    """ 
    Squeeze net model
    """
    img_input = Input(shape=INPUT_SHAPE)

    x = Conv2D(64, (3, 3), activation='elu', strides=(2, 2), padding='valid')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire(x, squeeze=16, expand=16)
    x = fire(x, squeeze=16, expand=16)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire(x, squeeze=32, expand=32)
    x = fire(x, squeeze=32, expand=32)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire(x, squeeze=48, expand=48)
    x = fire(x, squeeze=48, expand=48)
    #x = fire(x, squeeze=64, expand=64)
    #x = fire(x, squeeze=64, expand=64)
    #x = Dropout(0.5)(x)

    x = Conv2D(5, (1, 1), activation='elu', padding='valid')(x)
    x = Flatten()(x)

    out = Dense(1, activation='linear')(x)

    model= Model(img_input, out)

    model.summary()
    return model




