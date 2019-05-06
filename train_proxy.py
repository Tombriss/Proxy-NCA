from model_proxy3 import JointNet
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
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
# os.environ['OMP_NUM_THREADS']='8'


if __name__ == '__main__':

    start = timeit.default_timer()
    BATCH_SIZE = 32
    AUD_FEAT = 2048

    jointnet = JointNet()
    model = jointnet.model
    # model.load_weights('/home/data1/anshulg/triplet_relu_newdata_deep2_orth_drop_newdata.keras', by_name=True)
    # plot_model(model, 'proxymodel.png')
    # input()


    with open('/home/data1/anshulg/speech_features_2048D.pkl', 'rb') as fp:
        speech_data = pickle.load(fp)

    # max_len = speech_data['abacus'].shape[1]


    with open('/home/data1/kiranp/extracted_features/imagenet_xception/data_kmeans_train.pkl', 'rb') as fp:
        img_data_train = pickle.load(fp)

    # with open('/home/data1/kiranp/extracted_features/imagenet/data_kmeans_val.pkl', 'rb') as fp:
    #     img_data_val = pickle.load(fp)
    classes = list(img_data_train.keys())
    label_ind = {c:i for i,c in enumerate(classes)}

    df_train = pd.read_csv('/home/anshulg/WordNet/get_imagenet/train_data_proxy.csv')
    df_val = pd.read_csv('/home/anshulg/WordNet/get_imagenet/val_data_proxy.csv')

    
    # with open('/home/anshulg/WordNet/get_imagenet/proxy_aud.pkl', 'rb') as fp:
    #     proxy_aud = pickle.load(fp)

    # with open('/home/anshulg/WordNet/get_imagenet/proxy_img.pkl', 'rb') as fp:
    #     proxy_img = pickle.load(fp)

   

    def generator(df, img_data):
        
        num_samples = df.shape[0]
        i = 0
        while True:

            if i%num_samples==0:
                df = df.sample(frac=1).reset_index(drop=True)                
                i = 0
                
        
            groundings = df['0'].values[i:i+BATCH_SIZE]
            anchor_labels = df['1'].values[i:i+BATCH_SIZE]
            anchor_indices = df['2'].values[i:i+BATCH_SIZE]
            

            anchor_img = []
            class_mask = np.zeros((BATCH_SIZE, 576))
            class_mask_bar = np.ones((BATCH_SIZE, 576))
            anchor_aud = []
            for s,x in enumerate(groundings):                
                label = anchor_labels[s]
                class_mask[s][label_ind[label]] = 1
                class_mask_bar[s][label_ind[label]] = 0

                # image anchor
                if x==0:                    
                    anchor_img.append(img_data[label][anchor_indices[s]])
                    anchor_aud.append(np.zeros(AUD_FEAT))
                   
                    
                # audio anchor
                else:                   
                    anchor_aud.append(speech_data[label][anchor_indices[s]])
                    anchor_img.append(np.zeros(2048))

                               
            yield [groundings, 1-groundings, np.array(anchor_aud), np.array(anchor_img), np.array(class_mask), np.array(class_mask_bar)], np.zeros(BATCH_SIZE)
            
            i += BATCH_SIZE



    filepath = "/home/data1/anshulg/proxy_loss_deep2_actual_newdata.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
    callbacks_list = [checkpoint]


    # NOTE: val set consists of seen images and unseen audio so we are repeating img_data_train for val generator

    num_train_batches = (df_train.shape[0] / BATCH_SIZE)
    num_val_batches = (df_val.shape[0] / BATCH_SIZE)
    history = model.fit_generator(generator(df_train, img_data_train), steps_per_epoch=num_train_batches, epochs=400, callbacks=callbacks_list, validation_data=generator(df_val, img_data_train), validation_steps=num_val_batches)


    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig("proxy_loss_deep2_actual_newdata.png", bbox_inches = 'tight', pad_inches = 0)
    plt.close()

   
    with open('/home/data1/anshulg/history_proxy_loss_deep2_actual_newdata.json', 'w') as fp:
        json.dump(history.history, fp)

    stop = timeit.default_timer()
    time = open('time.txt','w')
    time.write(str(stop-start))
    