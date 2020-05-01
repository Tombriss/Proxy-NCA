from model_proxy import JointNet
import keras
import numpy as np
from keras.utils import plot_model
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import timeit
import json
import pandas as pd
from data_utils import DataSet
import warnings


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        plt.clf()
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.savefig("learning_curves.png", bbox_inches='tight', pad_inches=0)
        plt.close()


class ModelsSaver(keras.callbacks.Callback):

    def __init__(self, filepath, embbeder, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelsSaver, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.embbeder = embbeder

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                            self.embbeder.save_weights(filepath + '_embedder_', overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                            self.embbeder.save(filepath + '_embedder_', overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                    self.embbeder.save_weights(filepath + '_embedder_', overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
                    self.embbeder.save(filepath + '_embedder_', overwrite=True)


if __name__ == '__main__':

    start = timeit.default_timer()

    data_loader = DataSet()

    jointnet = JointNet(ip_size=data_loader.size, nbre_classes=data_loader.num_classes, lr=data_loader.lr)
    model = jointnet.model

    train_generator = data_loader.generator(mode='train')
    num_steps_train = data_loader.num_steps(mode='train')

    val_generator = data_loader.generator(mode='val')
    num_steps_val = data_loader.num_steps(mode='val')

    filepath = "model"
    checkpoint = ModelsSaver(filepath, jointnet.embedder, monitor='val_loss',
                             verbose=0, save_best_only=True, mode='min', period=1)
    plot_losses = PlotLosses()
    callbacks_list = [checkpoint, plot_losses]

    history = model.fit_generator(train_generator, steps_per_epoch=num_steps_train, epochs=data_loader.epochs,
                                  callbacks=callbacks_list, validation_data=val_generator, validation_steps=num_steps_val)

    with open('history_training.json', 'w') as fp:
        json.dump(history.history, fp)

    stop = timeit.default_timer()
    with open('time.txt', 'w') as f:
        f.write(str(stop - start))
