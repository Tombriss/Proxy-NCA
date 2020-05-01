import keras
import keras.backend as K
from keras.models import Model, load_model, Sequential
from keras.layers import Layer, Dense, Input, Dropout, Multiply, Lambda, Add, Flatten, Masking, LSTM, Conv2D, BatchNormalization, Activation, TimeDistributed
import tensorflow as tf
import numpy as np
from wrn import create_wide_residual_network


class JointNet():

    def __init__(self, ip_size, nbre_classes, is_test=True, lr=0.1):

        self.is_test = is_test

        self.input_size = ip_size
        self.nbre_classes = nbre_classes
        self.lr = lr
        self.image_submodel = self._image_submodel()
        self.image_submodel.summary()
        self.model = self.joint_model()
        self.model.summary()
        self.embedder = self.create_embedder()

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
        x = proxy_mat(anchor_latent)
        exp_ = K.exp(-2 * x)

        d_pos = K.sum(exp_ * class_mask)
        d_neg = K.sum(exp_ * class_mask_bar)

        loss = - K.log(d_pos / d_neg + 1e-16)

        return loss

    def _image_submodel(self):

        size = (self.input_size[0], self.input_size[1], 3,)

        if self.is_test:

            model = Sequential(name='sequential_2')
            model.add(Flatten(input_shape=size))
            model.add(Dense(64, activation='relu'))

        else:
            model = create_wide_residual_network(size, nb_classes=64, N=4, k=2, dropout=0)

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
        model.compile(loss=self.identity_loss, optimizer=keras.optimizers.Adam(lr=self.lr))

        return model

    def create_embedder(self):

        embedder_input = Input((self.input_size[0], self.input_size[1], 3, ), name='embedder_input')
        embedding = self.image_submodel(embedder_input)

        embedder = Model(
            inputs=embedder_input,
            outputs=embedding)

        embedder.compile(loss=self.identity_loss, optimizer=keras.optimizers.Adam(lr=self.lr))

        return(embedder)


if __name__ == '__main__':

    net = JointNet(ip_size=[32, 32], nbre_classes=19, lr=0.1)
