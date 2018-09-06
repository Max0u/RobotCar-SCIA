import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from d_nvi_utils import INPUT_SHAPE, batch_generator
import argparse
import os

from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """

    X = [f for f in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, f))]
    y = [(float(f.split("_")[3]), float(f.split("_")[5].split(".j")[0])) for f in X]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    Modified NVIDIA model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))

    image_input = Input(shape=INPUT_SHAPE)
    x = model(image_input)


    angle_out = Dense(1, name='angle_out')(x)
    speed_out = Dense(1, name='speed_out')(x)

    model = Model(inputs=[image_input], outputs=[angle_out, speed_out])
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    direct = 'd_nvi_models' 
    if not os.path.exists(direct):
        os.makedirs(direct)
    checkpoint = ModelCheckpoint(direct + '/model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss={'angle_out': root_mean_squared_error,
                        'speed_out': root_mean_squared_error},
                        loss_weights={'angle_out': 0.9, 'speed_out': .001},
                        optimizer=Adam(lr=args.learning_rate))

    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        steps_per_epoch=args.samples_per_epoch*2//args.batch_size,
                        epochs=args.nb_epoch,
                        max_queue_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        validation_steps=len(X_valid)*2//args.batch_size,
                        callbacks=[checkpoint],
                        verbose=1)


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',
            type=int,   default=20)
    parser.add_argument('-s', help='samples per epoch',
            dest='samples_per_epoch', type=int,   default=100000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',
            type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    #save_model_to_json(model)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()
