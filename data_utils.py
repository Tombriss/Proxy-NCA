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


class DataSet(object):
    """Dataset object that produces augmented training and eval data."""

    def __init__(self):

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

        mean = np.mean(train_data, axis=(0, 1, 2))
        std = np.std(train_data, axis=(0, 1, 2))
        self.mean_dataset = mean
        self.std_dataset = std

        all_data = (all_data - mean) / std
        all_labels = np.eye(num_classes)[np.array(all_labels, dtype=np.int32)]
        val_data = (val_data - mean) / std
        val_labels = np.eye(num_classes)[np.array(val_labels, dtype=np.int32)]
        train_data = (train_data - mean) / std
        train_labels = np.eye(num_classes)[np.array(train_labels, dtype=np.int32)]
        print('\n All images shape:{}, All labels shape:{}'.format(all_data.shape, all_labels.shape))
        assert len(all_data) == len(all_labels)
        assert len(train_data) == len(train_labels)
        assert len(val_data) == len(val_labels)

        print('Distribution of dataset over train and val:')
        pprint.pprint({
            label: {
                'train': len(dic_train_val['train']),
                'val': len(dic_train_val['val'])
            } for label, dic_train_val in train_val_per_label.items()}
        )

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

        print('\nDataset created')

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

    def generator(self, mode='train'):

        images, labels, labels_bar = self.get_data(mode=mode)
        num_samples = images.shape[0]

        i = 0
        while True:

            if i + self.batch_size > num_samples:
                i = 0

            batched_data = [
                images[i:i + self.batch_size],
                labels[i:i + self.batch_size],
                labels_bar[i:i + self.batch_size]]

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
    generator = data_loader.generator()
