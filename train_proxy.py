from model_proxy import JointNet
import keras
import numpy as np
from keras.utils import plot_model
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import timeit
import json
import pandas as pd
from data_utils import DataSet
import warnings
import os
import tensorflow as tf
import tensorflow.python.keras.backend as K

os.environ['PYTHONHASHSEED'] = str(1)
np.random.seed(1)
tf.compat.v1.set_random_seed(1)

data_loader = DataSet()
wrd = data_loader.out_dir

jointnet = JointNet(ip_size=data_loader.size, nbre_classes=data_loader.num_classes,
                    lr=data_loader.lr, is_test=data_loader.is_test, embedding_size=data_loader.embedding_size)


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.log_scale = True

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        if (len(self.val_losses) > 1 and len(self.losses) > 1) and (
            self.val_losses[-1] is not None and self.losses[-1] is not None
        ) and (
            self.val_losses[-1] <= 0 or self.val_losses[-1] <= 0
        ):
            self.log_scale = False

        self.i += 1

        plt.clf()
        if self.log_scale:
            plt.yscale('log')
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.savefig(wrd + "learning_curves.png", bbox_inches='tight', pad_inches=0)
        plt.close()


class StepDecay():
    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery
        self.prec = None

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = max(self.initAlpha * (self.factor ** exp), 0.001)
        if self.prec == None:
            self.prec = alpha
        if self.prec != alpha:
            print('Learning rate decay : from {:f} to {:f}.'.format(self.prec, alpha))
            self.prec = alpha

        # return the learning rate
        return float(alpha)


os.environ['PYTHONHASHSEED'] = str(1)
np.random.seed(1)
tf.compat.v1.set_random_seed(1)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

start = timeit.default_timer()

model = jointnet.model

train_generator = data_loader.generator(mode='train')
num_steps_train = data_loader.num_steps(mode='train')

val_generator = data_loader.generator(mode='val')
num_steps_val = data_loader.num_steps(mode='val')

filepath = wrd + "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_weights_only=True,
                             verbose=1, save_best_only=True, mode='min', period=data_loader.validation_freq)
plot_losses = PlotLosses()

schedule = StepDecay(initAlpha=data_loader.lr, factor=0.60, dropEvery=100)
callbacks_list = [checkpoint, plot_losses, LearningRateScheduler(schedule)]

# for i in range(num_steps_train):

#     data = next(train_generator)[0]
#     print('batch :')
#     print(data[0].shape, data[1].shape, data[2].shape, data[3].shape)

history = model.fit_generator(train_generator, steps_per_epoch=num_steps_train, epochs=data_loader.epochs,
                              callbacks=callbacks_list, validation_data=val_generator, validation_steps=num_steps_val, validation_freq=data_loader.validation_freq)

stop = timeit.default_timer()
with open(wrd + 'time.txt', 'w') as f:
    f.write(str(stop - start))
