import keras
import keras.backend as K
from keras.models import Model, load_model, Sequential
from keras.layers import Layer, Dense, Input, Dropout, Multiply, Lambda, Add, SpatialDropout2D, Flatten, Masking, LSTM, Conv2D, BatchNormalization, Activation, TimeDistributed
from keras.regularizers import l2
import tensorflow as tf
import numpy as np
from wrn import create_wide_residual_network
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, average_precision_score, f1_score, accuracy_score
from keras.constraints import UnitNorm
import os
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

os.environ['PYTHONHASHSEED'] = str(1)
np.random.seed(1)
tf.compat.v1.set_random_seed(1)


def distance(x, y):

    return (1 + tf.math.reduce_sum(x * y, axis=0) / tf.norm(x, ord='euclidean')) / 2


class JointNet():

    def __init__(self, ip_size, nbre_classes, is_test=False, lr=0.1, embedding_size=64):

        self.is_test = is_test
        self.embedding_size = embedding_size
        self.input_size = ip_size
        self.nbre_classes = nbre_classes
        self.lr = lr
        self.image_submodel = self._image_submodel()
        self.image_submodel.summary()
        self.model = self.joint_model()
        self.model.summary()
        self.embedder = None
        self.proxy_mat = None

    def identity_loss(self, y_true, y_pred):

        return K.mean(y_pred)

    def bpr_nca_loss(self, X):

        x, class_mask, class_mask_bar, proxy_mat = X

        dist_x = lambda y: distance(x, y)

        dist_mat = tf.map_fn(dist_x, proxy_mat)

        max_proba = tf.math.reduce_max(dist_mat)

        d_pos = tf.tensordot(dist_mat, class_mask, 1)
        d_neg = tf.tensordot(dist_mat, class_mask_bar, 1) / (self.nbre_classes - 1)

        norm_x = tf.norm(x, ord='euclidean')

        proba_pos = max_proba * norm_x
        c = tf.math.reduce_sum(class_mask)

        conf_neg = -(1 - c) * tf.math.log(1 - norm_x + 1e-6)
        conf_pos = -c * tf.math.log(norm_x + 1e-6)

        loss_confidence = conf_neg + conf_pos

        loss_conf_pos = -c * tf.math.log(max_proba + 1e-6) - (1 - c) * tf.math.log(1 - max_proba + 1e-6)
        loss_pos_neg = -c * tf.math.log(proba_pos + 1e-6) - (1 - c) * tf.math.log(1 - proba_pos + 1e-6)

        loss_classification = tf.math.log(1e-6 + (d_neg + 1e-16) / (d_pos + 1e-16))

        loss = 1 + loss_classification * c + 0.01 * loss_confidence + 0.01 * loss_conf_pos + 0.01 * loss_pos_neg

        return 1 + loss_classification * c

    def _image_submodel(self):

        size = (self.input_size[0], self.input_size[1], 3,)

        if self.is_test:

            p = 0
            reg = l2(0)

            inp = Input(size, name='first_layer')
            x = Conv2D(32, 3, padding="same", activation="relu", strides=2, input_shape=size, kernel_regularizer=reg)(inp)
            x = SpatialDropout2D(p)(x)
            x = Conv2D(64, 3, padding="same", activation="relu", strides=2, kernel_regularizer=reg)(x)
            x = SpatialDropout2D(p)(x)
            x = Conv2D(128, 3, padding="same", activation="relu", strides=2, kernel_regularizer=reg)(x)
            x = Flatten()(x)
            x = Dropout(p)(x)
            x = Dense(256, activation=None, kernel_regularizer=reg)(x)
            regularise = lambda x: x / 10
            x = Lambda(regularise)(x)
            x = Dense(self.embedding_size, activation='tanh')(x)
            normalize = lambda x: x / (self.embedding_size**0.5)
            x = Lambda(normalize, name='last_layer')(x)

            model = Model(inputs=inp, outputs=x, name='embedder')

        else:
            model = create_wide_residual_network(size, nb_classes=self.embedding_size, N=4, k=2, dropout=0.5)

        return model

    def joint_model(self):

        anchor_img = Input((self.input_size[0], self.input_size[1], 3, ), name='embedder_input')
        class_mask = Input((self.nbre_classes, ), name='class_mask')
        class_mask_bar = Input((self.nbre_classes, ), name='class_mask_bar')

        id_mat = Input((self.nbre_classes, self.nbre_classes, ), name='identity_matrix')

        proxy_mat = Dense(self.embedding_size, input_shape=(self.nbre_classes, self.nbre_classes),
                          use_bias=False, kernel_constraint=UnitNorm(axis=1), name='proxy_mat')(id_mat)

        anchor_img_latent = self.image_submodel(anchor_img)

        loss_computing = lambda X: tf.map_fn(self.bpr_nca_loss, X, dtype=tf.float32)
        loss_layer = Lambda(loss_computing, output_shape=(1, ), name='loss')
        loss = loss_layer([anchor_img_latent, class_mask, class_mask_bar, proxy_mat])

        model = Model(
            inputs=[anchor_img, class_mask, class_mask_bar, id_mat],
            outputs=loss)
        model.compile(loss=self.identity_loss, optimizer=keras.optimizers.Adam(lr=self.lr))

        return model

    def load_weights(self, path='model.h5'):

        self.model.load_weights(path)

        embedder_input = self.model.get_layer('embedder').get_layer('first_layer').output
        embedder_output = self.model.get_layer('embedder').get_layer('last_layer').output
        self.embedder = Model(inputs=embedder_input, outputs=embedder_output)

        proxy_mat_input = self.model.get_layer('identity_matrix').output
        proxy_mat_output = self.model.get_layer('proxy_mat').output
        self.proxy_mat = Model(inputs=proxy_mat_input, outputs=proxy_mat_output)

    def get_proxy_mat(self):

        if self.proxy_mat is None:
            self.load_weights()

        id_ = np.tile(np.identity(self.nbre_classes, dtype=np.float32), (1, 1, 1))

        return(self.proxy_mat.predict(id_)[0])

    def run_test(self):

        tf.compat.v1.enable_eager_execution()

        batch_size = 10
        self.embedding_size = 64
        image = np.random.random_sample((batch_size, self.embedding_size)) * 10
        # image = np.ones((batch_size, self.embedding_size)) * 10000000
        print(image)
        image = tf.math.tanh(image)
        print(image)
        alpha = 1 / (self.embedding_size**0.5)
        print(alpha)
        image = alpha * image
        print(image)
        print(tf.map_fn(lambda x: tf.norm(x, ord='euclidean'), image, dtype=tf.float64))
        # image = tf.map_fn(lambda x: tf.math.tanh(tf.norm(x, ord='euclidean'))
        #   * x / tf.norm(x, ord='euclidean'), image, dtype=tf.float64)
        # # print(tf.map_fn(lambda x: tf.norm(x, ord='euclidean'), image_2, dtype=tf.float64))
        class_mask = np.eye(self.nbre_classes)[np.random.choice(self.nbre_classes, 1)]
        print(class_mask.shape)
        # class_mask = np.concatenate([class_mask, np.zeros((5, self.nbre_classes))])
        class_mask_bar = np.ones((batch_size, self.nbre_classes)) - class_mask
        proxy_mat = np.tile(np.random.random_sample((self.nbre_classes, self.embedding_size)), (batch_size, 1, 1))
        sum_ = proxy_mat.sum(axis=1)
        proxy_mat = proxy_mat / sum_[:, np.newaxis]

        X = [image, class_mask, class_mask_bar, proxy_mat]

        print(tf.map_fn(self.bpr_nca_loss, X, dtype=tf.float64))

    def get_knn(self, data_loader, mode='enhanced', negs=True):

        if self.proxy_mat is None:
            self.load_weights()

        generator = data_loader.generator(mode='train', augment=False)
        num_steps = data_loader.num_steps(mode='train')
        list_X, list_y = [], []

        proxy_mat = self.get_proxy_mat()

        for _ in range(num_steps):
            batch = next(generator)
            list_X.append(self.embedder.predict(batch[0][0]))
            list_y.append(batch[0][1])

        X = np.concatenate(list_X)
        y = np.concatenate(list_y)

        if mode == 'enhanced':

            n_neighbors = 6
            X_tot = np.concatenate([X, proxy_mat])
            y_tot = np.concatenate([y, np.identity(self.nbre_classes)])
            y_tot = np.array([(np.where(r == 1)[0][0] + 1 if len(np.where(r == 1)[0]) == 1 else 0) for r in y_tot])

        elif mode == 'honest':

            n_neighbors = 6
            X_tot = np.concatenate([X, proxy_mat])
            y_tot = np.concatenate([y, np.identity(self.nbre_classes)])

            y_tot = np.array([(np.where(r == 1)[0][0] + 1 if len(np.where(r == 1)[0]) == 1 else 0) for r in y_tot])

            X_tot = X_tot[y_tot != 0]
            y_tot = y_tot[y_tot != 0]

            X_tot = np.concatenate([X_tot, np.zeros((6, self.embedding_size))])
            y_tot = np.concatenate([y_tot, np.zeros((6,))])

        else:
            n_neighbors = 1
            X_tot = np.concatenate([np.zeros((1, self.embedding_size)), proxy_mat])
            y_tot = np.array(list(range(self.nbre_classes + 1)))

        knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_tot, y_tot)

        X_pos = X_tot[y_tot != 0]
        y_pos = y_tot[y_tot != 0]

        knn_pos = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_pos, y_pos)

        knn_neg = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_tot, np.array([(0 if i == 0 else 1)for i in y_tot]))

        return(knn, knn_pos, knn_neg)

    def predict(self, images, data_loader, mode='enhanced'):

        if self.proxy_mat is None:
            self.load_weights()

        X = self.embedder.predict(images)

        knn = self.get_knn(data_loader, mode=mode)
        knn_predictions = knn.predict(X)

        return(knn_predictions)

    def _evaluate_batch(self, X, labels, knn, name='', data_loader=None, plot=False, hist=True, rec_prec_curves=False):

        if self.proxy_mat is None:
            self.load_weights()

        n_classes_plot = 6

        knn_predictions = knn.predict(X)

        if plot:

            tsne = TSNE(n_components=2, verbose=1, n_jobs=-1)
            tsne_results = tsne.fit_transform(X)

            # 'centrum_women':30,'centrum_women':55

            df = pd.DataFrame({'tsne-2d-one': tsne_results[:, 0], 'tsne-2d-two': tsne_results[:, 1], 'y': labels})
            df['label'] = df['y'].map(data_loader.label_map)
            df_subset = df[df['y'].isin([0, 43, 31, 56, 41, 48])]  # set_lbl[:n_classes_plot - 4]

            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x="tsne-2d-one", y="tsne-2d-two",
                hue="label",
                # palette=sns.color_palette("hls", n_classes_plot),
                data=df_subset,
                legend="full",
                alpha=0.3
            )
            plt.savefig("tsne_{}.png".format(name), bbox_inches='tight', pad_inches=0)

        proxy_mat = self.get_proxy_mat()
        probs = []
        sums = []
        for z in X:
            max_proba_z = None
            sum_z = 0
            for p in proxy_mat:
                proba_z_p = (1 + np.sum(p * z, axis=0) / np.linalg.norm(z)) / 2
                if max_proba_z is None or proba_z_p > max_proba_z:
                    max_proba_z = proba_z_p
                sum_z += proba_z_p
            sums.append(sum_z / self.nbre_classes)
            probs.append(max_proba_z)

        probs = np.array(probs)

        if hist:

            p_neg = probs[labels == 0]
            p_pos = probs[labels != 0]

            h_neg = np.sum(np.abs(X[labels == 0])**2, axis=-1)**(1. / 2)
            h_pos = np.sum(np.abs(X[labels != 0])**2, axis=-1)**(1. / 2)

            prod_neg = p_neg * h_neg
            prod_pos = p_pos * h_pos

            bins = np.linspace(0, 1, 100)

            plt.figure()

            plt.hist(h_neg, bins, alpha=0.5, label='norms of negs', density=True)
            plt.hist(h_pos, bins, alpha=0.5, label='norms of pos', density=True)
            plt.legend(loc='upper right')
            plt.savefig("hist_norms_{}.png".format(name), bbox_inches='tight', pad_inches=0)

            plt.figure()

            plt.hist(p_neg, bins, alpha=0.5, label='probs of negs', density=True)
            plt.hist(p_pos, bins, alpha=0.5, label='probs of pos', density=True)
            plt.legend(loc='upper right')
            plt.savefig("hist_probs_{}.png".format(name), bbox_inches='tight', pad_inches=0)

            plt.figure()

            plt.hist(prod_neg, bins, alpha=0.5, label='prods of negs', density=True)
            plt.hist(prod_pos, bins, alpha=0.5, label='prods of pos', density=True)
            plt.legend(loc='upper right')
            plt.savefig("hist_prods_{}.png".format(name), bbox_inches='tight', pad_inches=0)

        if name == 'val' and rec_prec_curves:

            norms = np.sum(np.abs(X)**2, axis=-1)**(1. / 2)
            plt.figure()
            precision1, recall1, thresholds1 = precision_recall_curve(labels, norms)
            f11 = 2 / (1 / np.array(precision1[:-1]) + 1 / np.array(recall1[:-1]))
            plt.plot(thresholds1, precision1[:-1], 'b', label='precision')
            plt.plot(thresholds1, recall1[:-1], 'g', label='recall')
            plt.plot(thresholds1, f11, 'r', label='f1')
            plt.legend(loc='lower left')
            plt.savefig("pre_rec_norms_{}.png".format(name), bbox_inches='tight', pad_inches=0)

            plt.figure()
            precision2, recall2, thresholds2 = precision_recall_curve(labels, probs)
            f12 = 2 / (1 / np.array(precision2[:-1]) + 1 / np.array(recall2[:-1]))
            plt.plot(thresholds2, precision2[:-1], 'b', label='precision')
            plt.plot(thresholds2, recall2[:-1], 'g', label='recall')
            plt.plot(thresholds2, f12, 'r', label='f1')
            plt.legend(loc='lower left')
            plt.savefig("pre_rec_probs_{}.png".format(name), bbox_inches='tight', pad_inches=0)

            prods = norms * probs

            plt.figure()
            precision3, recall3, thresholds3 = precision_recall_curve(labels, prods)
            f13 = 2 / (1 / np.array(precision3[:-1]) + 1 / np.array(recall3[:-1]))
            plt.plot(thresholds3, precision3[:-1], 'b', label='precision')
            plt.plot(thresholds3, recall3[:-1], 'g', label='recall')
            plt.plot(thresholds3, f13, 'r', label='f1')
            plt.legend(loc='lower left')
            plt.savefig("pre_rec_prods_{}.png".format(name), bbox_inches='tight', pad_inches=0)

            plt.figure()
            plt.plot(thresholds1, precision1[:-1], 'b', label='precision norms', linestyle='solid')
            plt.plot(thresholds1, recall1[:-1], 'g', label='recall norms', linestyle='solid')
            plt.plot(thresholds1, f11, 'r', label='f1 norms', linestyle='solid')
            plt.plot(thresholds2, precision2[:-1], 'b', label='precision probs', linestyle='dashed')
            plt.plot(thresholds2, recall2[:-1], 'g', label='recall probs', linestyle='dashed')
            plt.plot(thresholds2, f12, 'r', label='f1 probs', linestyle='dashed')
            plt.plot(thresholds3, precision3[:-1], 'b', label='precision prods', linestyle='dotted')
            plt.plot(thresholds3, recall3[:-1], 'g', label='recall prods', linestyle='dotted')
            plt.plot(thresholds3, f13, 'r', label='f1', linestyle='dotted')
            plt.legend(loc='lower left')
            plt.savefig("pre_rec_all_{}.png".format(name), bbox_inches='tight', pad_inches=0)

        # cm = confusion_matrix(labels, knn_predictions)
        ac = accuracy_score(labels, knn_predictions)
        ba = balanced_accuracy_score(labels, knn_predictions)
        f1 = f1_score(labels, knn_predictions, average='macro')

        return(ac, ba, f1)

    def evaluate(self, data_loader, knn_mode='enhanced', plot=False):

        knn, knn_pos, knn_neg = self.get_knn(data_loader, mode=knn_mode)

        for mode in ['train', 'val']:

            generator = data_loader.generator(mode=mode, augment=False)
            num_steps = data_loader.num_steps(mode=mode)
            list_X, list_y = [], []

            for _ in range(num_steps):
                batch = next(generator)
                list_X.append(self.embedder.predict(batch[0][0]))
                list_y.append(batch[0][1])

            X = np.concatenate(list_X)
            y = np.concatenate(list_y)
            y = np.array([(np.where(r == 1)[0][0] + 1 if len(np.where(r == 1)[0]) == 1 else 0) for r in y])

            ac, ba, f1 = self._evaluate_batch(X, y, knn, name=mode,
                                              data_loader=data_loader, plot=plot, hist=True)

            print('\n\nOn {} set:'.format(mode))
            print('\n all:')
            print("  accuracy : {0:5.2f}%".format(ac * 100))
            print("  balanced Accuracy : {0:5.2f}%".format(ba * 100))
            print("  f1 score : {0:5.2f}%".format(f1 * 100))

            ac, ba, f1 = self._evaluate_batch(X[y != 0], y[y != 0], knn_pos, name=mode,
                                              data_loader=data_loader, plot=False, hist=False)

            print('\n only positives:')
            print("  accuracy : {0:5.2f}%".format(ac * 100))
            print("  balanced Accuracy : {0:5.2f}%".format(ba * 100))
            print("  f1 score : {0:5.2f}%".format(f1 * 100))

            ac, ba, f1 = self._evaluate_batch(X, np.array([(0 if i == 0 else 1)for i in y]), knn_neg, name=mode,
                                              data_loader=data_loader, plot=False, hist=False, rec_prec_curves=True)

            print('\n negatives vs positives:')
            print('  (for a total of {} negatives and {} positives)'.format(len(y[y == 0]), len(y[y != 0])))
            print("  accuracy : {0:5.2f}%".format(ac * 100))
            print("  balanced Accuracy : {0:5.2f}%".format(ba * 100))
            print("  f1 score : {0:5.2f}%".format(f1 * 100))

            # print('Confusion_matrix for {} set:'.format(mode))
            # print(cm)


if __name__ == '__main__':

    net = JointNet(ip_size=[32, 32], nbre_classes=19, lr=0.1)
    net.run_test()
