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


if __name__ == '__main__':

    start = timeit.default_timer()

    data_loader = DataSet()

    jointnet = JointNet(ip_size=data_loader.size, nbre_classes=data_loader.num_classes)
    model = jointnet.model

    train_generator = data_loader.generator(mode='train')
    num_steps_train = data_loader.num_steps(mode='train')

    val_generator = data_loader.generator(mode='val')
    num_steps_val = data_loader.num_steps(mode='val')

    filepath = "model.ckpt"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
    callbacks_list = [checkpoint]

    history = model.fit_generator(train_generator, steps_per_epoch=num_steps_train, epochs=3,
                                  callbacks=callbacks_list, validation_data=val_generator, validation_steps=num_steps_val)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig("history.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    with open('history_training.json', 'w') as fp:
        json.dump(history.history, fp)

    stop = timeit.default_timer()
    with open('time.txt', 'w') as f:
        f.write(str(stop - start))
