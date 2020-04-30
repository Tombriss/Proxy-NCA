import keras
import keras.backend as K
from keras.models import Model, load_model, Sequential
from keras.layers import Layer, Dense, Input, Dropout, Multiply, Lambda, Add, Flatten, Masking, LSTM, Conv2D, BatchNormalization, Activation, TimeDistributed
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops

distance = lambda x: (K.sum(K.square(x[0] - x[1]), axis=-1, keepdims=True))


class JointNet():

    def __init__(self, ip_size, nbre_classes):

        self.input_size = ip_size
        self.nbre_classes = nbre_classes
        self.image_submodel = self._image_submodel()
        self.image_submodel.summary()
        self.model = self.joint_model()
        self.model.summary()

    def identity_loss(self, y_true, y_pred):

        return K.mean(y_pred)

    def bpr_nca_loss(self, X):

        anchor_latent, class_mask, class_mask_bar = X

        anchor_latent = K.l2_normalize(anchor_latent, axis=-1)
        proxy_mat = Dense(self.nbre_classes, input_shape=(64, ), use_bias=False, kernel_constraint='UnitNorm')

        # COSINE SIMILARITY

        # d_pos = K.exp(-distance([proxy_mat(anchor_latent), class_mask]))
        # d_neg = K.exp(-distance([proxy_mat(anchor_latent), class_mask_bar]))

        # loss = K.log(d_neg / d_pos + 1e-16)
        vec = 2 * class_mask_bar * proxy_mat(anchor_latent)

        ex = K.exp(vec)

        loss = K.log(K.sum(ex))

        return loss

    def _image_submodel(self):

        model = Sequential(name='sequential_2')
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(self.input_size[0], self.input_size[1], 3,)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', name='dense_img3'))

        return model

    def joint_model(self):

        anchor_img = Input((self.input_size[0], self.input_size[1], 3, ), name='anchor_img')
        class_mask = Input((self.nbre_classes, ), name='class_mask')
        class_mask_bar = Input((self.nbre_classes, ), name='class_mask_bar')

        anchor_img_latent = self.image_submodel(anchor_img)

        loss_layer = Lambda(self.bpr_nca_loss, output_shape=(1, ), name='loss')
        loss = loss_layer([anchor_img_latent, class_mask, class_mask_bar])

        model = Model(
            inputs=[anchor_img, class_mask, class_mask_bar],
            outputs=loss)
        model.compile(loss=self.identity_loss, optimizer=keras.optimizers.Adam(lr=0.001))

        return model


if __name__ == '__main__':

    net = JointNet(ip_size=[32, 32], nbre_classes=19)
