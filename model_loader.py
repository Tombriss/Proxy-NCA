import keras
import keras.backend as K
import keras.losses
from keras.models import Model
from data_utils import DataSet
from model_proxy import JointNet
import numpy as np
import os
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = str(1)
np.random.seed(1)
tf.compat.v1.set_random_seed(1)


if __name__ == '__main__':

    data_loader = DataSet()
    jointnet = JointNet(ip_size=data_loader.size, nbre_classes=data_loader.num_classes,
                        lr=data_loader.lr, is_test=data_loader.is_test, embedding_size=data_loader.embedding_size)

    jointnet.load_weights('model.h5')
    jointnet.evaluate(data_loader, knn_mode='enhanced', plot=True)

    # proxy_mat = jointnet.get_proxy_mat()
    # print(proxy_mat)

    # generator = data_loader.generator(mode='val')
    # image_test = (next(generator)[0][0]).reshape(85, data_loader.size[0], data_loader.size[1], 3)
    # print(jointnet.embedder.predict(image_test))
    # print(jointnet.get_proxy_mat())
