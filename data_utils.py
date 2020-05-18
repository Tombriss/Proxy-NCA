from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import numpy as np
import tensorflow as tf
import math
import time
import pprint
import json
import pandas as pd
from PIL import Image
import os

import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


class DataSet(object):
    """Dataset object that produces augmented training and eval data."""

    def __init__(self):

        os.environ['PYTHONHASHSEED'] = str(1)
        np.random.seed(1)
        tf.compat.v1.set_random_seed(1)

        print('\nGetting dataset...')

        with open("config_BO.json", 'r') as f:
            config_BO = json.load(f)

        self.size = config_BO["size"]
        self.source = config_BO["catalog_path"]
        self.max_classes = config_BO["max_classes"]
        self.max_sample_per_classes = config_BO["max_sample_per_classes"]
        self.max_validation_sample_per_classes = config_BO["max_validation_sample_per_classes"]
        self.max_training_sample_per_classes = config_BO["max_training_sample_per_classes"]
        self.min_sample_per_classes = config_BO["min_sample_per_classes"]
        self.batch_size = config_BO["batch_size"]
        self.split_factor = config_BO["split_factor"]

        self.epochs = config_BO["epochs"]
        self.lr = config_BO["lr"]
        self.is_test = config_BO["is_test"]
        self.embedding_size = config_BO["embedding_size"]
        self.validation_freq = config_BO["validation_freq"]

        self.source_neg = config_BO["negative_path"]
        self.n_neg_train = config_BO["n_neg_train"]
        self.n_neg_test = config_BO["n_neg_test"]

        self.curr_train_index = 0

        label_map = []
        dataset_per_label = {}

        for root, _, files in os.walk(self.source):

            if len(files) < self.min_sample_per_classes:
                continue

            class_label = os.path.relpath(root, self.source)

            if(class_label == '.'):
                continue

            if class_label not in label_map:

                if len(label_map) >= self.max_classes:
                    break

                label_map.append(class_label)

            for file in files:
                if file == ".DS_Store":
                    continue
                file_path = os.path.join(class_label, file)
                lbl = label_map.index(class_label)
                if lbl in dataset_per_label:

                    if len(dataset_per_label[lbl]) >= self.max_sample_per_classes:
                        break

                    dataset_per_label[lbl].append(file_path)
                else:
                    dataset_per_label[lbl] = [file_path]

        self.label_map = dict(zip(list(range(1, len(label_map) + 1)), label_map))
        self.label_map.update({0: 'neg'})

        pprint.pprint(self.label_map)

        train_val_per_label = dict.fromkeys(range(len(dataset_per_label)), None)
        for label, images in dataset_per_label.items():

            label_train_size = int(len(images) * self.split_factor) + 1
            label_train_size = min(label_train_size, self.max_training_sample_per_classes)

            train_images = images[:label_train_size]
            end_val = min(label_train_size + self.max_validation_sample_per_classes, len(images))
            val_images = images[label_train_size:end_val]

            if len(val_images) == 0 and len(train_images) > 1:
                val_images = [train_images[-1]]
                train_images.pop()

            train_val_per_label[label] = {}

            train_val_per_label[label]['train'] = train_images
            train_val_per_label[label]['val'] = val_images

        self.train_size = sum(len(dic_label['train']) for dic_label in train_val_per_label.values())
        self.val_size = sum(len(dic_label['val']) for dic_label in train_val_per_label.values())
        self.dataset_size = self.train_size + self.val_size

        train_data = np.empty((self.train_size, self.size[0], self.size[1], 3), dtype=np.uint8)
        val_data = np.empty((self.val_size, self.size[0], self.size[1], 3), dtype=np.uint8)
        all_data = np.empty((self.dataset_size, self.size[0], self.size[1], 3), dtype=np.uint8)

        train_labels = [None] * self.train_size
        val_labels = [None] * self.val_size
        all_labels = [None] * self.dataset_size

        train_index = 0
        val_index = 0
        all_index = 0

        for label, train_val_dict in train_val_per_label.items():

            for image in train_val_dict['train']:

                img_array = self.name_to_array(image, self.source, self.size)

                train_data[train_index] = img_array
                train_labels[train_index] = label

                all_data[all_index] = img_array
                all_labels[all_index] = label

                train_index += 1
                all_index += 1

            for image in train_val_dict['val']:

                img_array = self.name_to_array(image, self.source, self.size)

                val_data[val_index] = img_array
                val_labels[val_index] = label

                all_data[all_index] = img_array
                all_labels[all_index] = label

                val_index += 1
                all_index += 1

        all_data = all_data / 255.0
        val_data = val_data / 255.0
        train_data = train_data / 255.0

        num_classes = len(label_map)
        self.num_classes = num_classes

        negs = self.get_negatives()

        train_negs = np.empty((self.n_neg_train, self.size[0], self.size[1], 3), dtype=np.uint8)
        val_negs = np.empty((self.n_neg_test, self.size[0], self.size[1], 3), dtype=np.uint8)

        train_labels_negs = np.zeros((self.n_neg_train, num_classes))
        val_labels_negs = np.zeros((self.n_neg_test, num_classes))

        train_index = 0
        val_index = 0

        for image in negs['train']:

            img_array = self.name_to_array(image, self.source_neg, self.size)
            train_negs[train_index] = img_array
            train_index += 1

        for image in negs['val']:

            img_array = self.name_to_array(image, self.source_neg, self.size)
            val_negs[val_index] = img_array
            val_index += 1

        train_negs = train_negs / 255.0
        val_negs = val_negs / 255.0

        mean = np.mean(train_data, axis=(0, 1, 2))
        std = np.std(train_data, axis=(0, 1, 2))
        self.mean_dataset = mean
        self.std_dataset = std

        train_data = np.concatenate([train_data, train_negs])
        val_data = np.concatenate([val_data, val_negs])

        val_labels = np.eye(num_classes)[np.array(val_labels, dtype=np.int32)]
        train_labels = np.eye(num_classes)[np.array(train_labels, dtype=np.int32)]

        val_labels = np.concatenate([val_labels, val_labels_negs])
        train_labels = np.concatenate([train_labels, train_labels_negs])

        val_data = (val_data - mean) / std
        train_data = (train_data - mean) / std

        self.train_size = len(train_data)
        self.val_size = len(val_data)

        assert len(train_data) == len(train_labels)
        assert len(val_data) == len(val_labels)

        print('Distribution of dataset over train and val:')
        pprint.pprint({
            label: {
                'train': len(dic_train_val['train']),
                'val': len(dic_train_val['val'])
            } for label, dic_train_val in train_val_per_label.items()}
        )

        pprint.pprint({'negs train': len(negs['train']), 'negs val': len(negs['val'])})

        self.train_images = train_data
        self.train_labels = train_labels

        self.val_images = val_data
        self.val_labels = val_labels

        self.num_train = self.train_images.shape[0]
        self.num_val = self.val_images.shape[0]

        assert self.train_images.shape[0] == len(self.train_labels)
        assert self.val_images.shape[0] == len(self.val_labels)
        assert self.batch_size <= self.num_train
        assert self.batch_size <= self.num_val

        aug_factor = 0.05
        self.seq = iaa.Sometimes(p=0.2, then_list=[iaa.Sequential([
            # iaa.Fliplr(0.1),
            # iaa.Fliplr(0.1),  # horizontal flips
            iaa.Crop(percent=(0, aug_factor)),  # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            # iaa.Sometimes(
            #     0.5,
            #     iaa.GaussianBlur(sigma=(0, 0.5))
            # ),
            # Strengthen or weaken the contrast in each image.
            # iaa.LinearContrast((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((1 - aug_factor, 1 + aug_factor), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (1 - aug_factor, 1 + aug_factor), "y": (1 - aug_factor, 1 + aug_factor)},
                translate_percent={"x": (-aug_factor, aug_factor), "y": (-aug_factor, aug_factor)},
                rotate=(-aug_factor, aug_factor),  # (-25, 25)
                shear=(-aug_factor, aug_factor)  # (-8, 8)
            )
        ], random_order=True)])  # apply augmenters in random order

        print('\nDataset created')

    def get_negatives(self):

        label_map = []
        dataset_per_label = {}

        for root, _, files in os.walk(self.source_neg):

            if len(files) < 0:
                continue

            class_label = os.path.relpath(root, self.source_neg)

            if(class_label == '.'):
                continue

            if class_label not in label_map:

                if len(label_map) >= 10000:
                    break

                label_map.append(class_label)

            for file in files:
                if file == ".DS_Store":
                    continue
                file_path = os.path.join(class_label, file)
                lbl = label_map.index(class_label)
                if lbl in dataset_per_label:

                    if len(dataset_per_label[lbl]) >= 10000:
                        break

                    dataset_per_label[lbl].append(file_path)
                else:
                    dataset_per_label[lbl] = [file_path]

        n = len(label_map) // 2
        train_neg_lbl = list(range(0, n))
        print('number of negative classes in test set:', n)
        test_neg_lbl = list(range(n, len(label_map)))

        max_length_class = self.n_neg_train // n
        train_images = []
        val_images = []
        n_to_add = self.n_neg_train % n
        for lbl in train_neg_lbl:
            length_class = min(max_length_class + n_to_add, len(dataset_per_label[lbl]))
            n_to_add = max_length_class + n_to_add - length_class
            train_images += dataset_per_label[lbl][:length_class]
        for lbl in train_neg_lbl:
            for im in dataset_per_label[lbl][::-1]:
                if n_to_add == 0:
                    break
                if im in train_images:
                    break
                else:
                    train_images.append(im)
                    n_to_add -= 1
        n_to_add = self.n_neg_train % n
        max_length_class = self.n_neg_test // n
        for lbl in test_neg_lbl:
            length_class = min(max_length_class + n_to_add, len(dataset_per_label[lbl]))
            n_to_add = max_length_class + n_to_add - length_class
            val_images += dataset_per_label[lbl][:length_class]

        pprint.pprint({label_map[lbl]: len(dataset_per_label[lbl]) for lbl in train_neg_lbl})
        pprint.pprint({label_map[lbl]: len(dataset_per_label[lbl]) for lbl in test_neg_lbl})

        np.random.shuffle(train_images)
        np.random.shuffle(val_images)

        self.n_neg_train = min(len(train_images), self.n_neg_train)
        self.n_neg_test = min(len(val_images), self.n_neg_test)
        negs = {}
        negs['train'] = train_images[:self.n_neg_train]
        negs['val'] = val_images[:self.n_neg_test]

        return(negs)

    @staticmethod
    def name_to_array(image, source, size):

        image_path = os.path.join(source, image)
        img = Image.open(image_path)
        img = img.resize(size)
        img_array = np.array(img)

        return(img_array)

    def num_steps(self, mode='train'):

        images = self.train_images if mode == 'train' else self.val_images
        nbre_samples = images.shape[0]

        return(nbre_samples // self.batch_size)

    def get_data(self, mode='train'):

        images = self.train_images if mode == 'train' else self.val_images
        labels = self.train_labels if mode == 'train' else self.val_labels
        nbre_samples = images.shape[0]
        labels_bar = np.ones((nbre_samples, self.num_classes)) - labels

        indexes = list(range(0, nbre_samples))
        np.random.shuffle(indexes)

        return([images[indexes], labels[indexes], labels_bar[indexes]])

    def generator(self, mode='train', augment=True):

        images, labels, labels_bar = self.get_data(mode=mode)
        num_samples = images.shape[0]
        identity_mats = np.tile(np.identity(self.num_classes, dtype=np.float32), (self.batch_size, 1, 1))

        i = 0
        while True:

            if i + self.batch_size > num_samples:
                i = 0

            batch_images = self.seq(images=images[i:i + self.batch_size].astype(np.float32)
                                    ) if (mode == 'train' and augment) else images[i:i + self.batch_size]

            batched_data = [
                batch_images,
                labels[i:i + self.batch_size],
                labels_bar[i:i + self.batch_size],
                identity_mats]

            yield batched_data, np.zeros(self.batch_size)

            i += self.batch_size

    def next_batch(self, epoch, batch_nbr, test=False):
        """Return the next minibatch of augmented data."""
        next_train_index = self.curr_train_index + self.batch_size
        if next_train_index > self.num_train:
            # Increase epoch number
            epoch = self.epochs + 1
            self.reset()
            self.epochs = epoch
        batched_data = (
            self.train_images[self.curr_train_index:
                              self.curr_train_index + self.batch_size],
            self.train_labels[self.curr_train_index:
                              self.curr_train_index + self.batch_size])
        images, labels = batched_data
        batched_data = (np.array(images, np.float32), labels)
        self.curr_train_index += self.batch_size
        return batched_data

    def reset(self):
        """Reset training data and index into the training data."""
        self.epochs = 0
        self.curr_train_index = 0


if __name__ == '__main__':

    data_loader = DataSet()
