import keras
import keras.backend as K
from keras.models import Model, load_model, Sequential
#from keras.applications.vgg16 import VGG16
from keras.layers import Layer, Dense, Input, Dropout, merge, Multiply, Add, Masking, LSTM, BatchNormalization, Activation, TimeDistributed
import tensorflow as tf
import numpy as np



class JointNet():

    def __init__(self):

        self.audio_submodel = self.audio_submodel()
        self.audio_submodel.summary()
        self.image_submodel = self.image_submodel()
        self.image_submodel.summary()
        self.model = self.joint_model()
        self.model.summary()


    # def __call__(self, model_option):
    #     if model_option == 'train':
    #         return self.model
    #     else:
    #         return None

    def identity_loss(self, y_true, y_pred):

        return K.mean(y_pred)



    def bpr_triplet_loss(self, X):

        anchor_latent, class_mask, class_mask_bar = X
            
               
        anchor_latent = K.l2_normalize(anchor_latent, axis=-1)
        proxy_mat = Dense(576, input_shape=(64, ), use_bias=False, kernel_constraint='UnitNorm')
                

        # COSINE SIMILARITY
        d_pos = K.exp(K.sum(proxy_mat(anchor_latent) * class_mask, axis=-1, keepdims=True))
        d_neg = K.exp(K.sum(proxy_mat(anchor_latent) * class_mask_bar, axis=-1, keepdims=True))
        

        loss = K.log(d_neg/d_pos + 1e-16)

        return loss



    def audio_submodel(self):

        input_size = 80
        hidden_size = 128

        model = Sequential(name='sequential_1')        
        model.add(Dense(512, activation='relu', input_shape=(2048, ), name='dense_aud1'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu', name='dense_aud2'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu', name='dense_aud3'))

        return model

    
    def image_submodel(self):

        model = Sequential(name='sequential_2')
        model.add(Dense(512, activation='relu', input_shape=(2048, ), name='dense_img1'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu', name='dense_img2'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu', name='dense_img3'))

        return model

    
    def joint_model(self):

        grounding = Input((1, ), name='grounding')
        grounding_bar = Input((1, ), name='grounding_bar')
        anchor_aud = Input((2048, ), name='anchor_aud')
        anchor_img = Input((2048, ), name='anchor_img')
        class_mask = Input((576, ), name='class_mask')
        class_mask_bar = Input((576, ), name='class_mask_bar')
        

        anchor_aud_latent = self.audio_submodel(anchor_aud)
        anchor_img_latent = self.image_submodel(anchor_img)
      

        anchor = Add()([Multiply()([grounding_bar, anchor_img_latent]), Multiply()([grounding, anchor_aud_latent])])     


        loss = merge(
            [anchor, class_mask, class_mask_bar],
            mode=self.bpr_triplet_loss,
            name='loss',
            output_shape=(1, ))

        model = Model(
            input=[grounding, grounding_bar, anchor_aud, anchor_img, class_mask, class_mask_bar],
            output=loss)
        model.compile(loss=self.identity_loss, optimizer=keras.optimizers.Adam(lr=0.01))

        return model