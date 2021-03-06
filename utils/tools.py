#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 11:36 2021

@author: Pedro Vieira
"""

import torch
import numpy as np
from scipy import io
from sklearn.preprocessing import StandardScaler
import random
import os
import glob
from tqdm import tqdm


# Dataset class
class HSIData:
    """Stores dataset raw image and labels and applies pre-processing"""

    def __init__(self, dataset_name, target_folder='./datasets/'):
        self.dataset_name = dataset_name
        folder = target_folder + dataset_name + '/'

        self.rgb_bands = None
        self.label_values = None
        if dataset_name == 'IndianPines':
            img = io.loadmat(folder + 'Indian_pines_corrected.mat')['indian_pines_corrected']
            gt = io.loadmat(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
            self.label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                                 "Corn", "Grass-pasture", "Grass-trees",
                                 "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                                 "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                                 "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                                 "Stone-Steel-Towers"]
            self.rgb_bands = (43, 21, 11)  # AVIRIS sensor
        elif dataset_name == 'PaviaU':
            img = io.loadmat(folder + 'PaviaU.mat')['paviaU']
            gt = io.loadmat(folder + 'PaviaU_gt.mat')['paviaU_gt']
            self.label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                                 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                                 'Self-Blocking Bricks', 'Shadows']
            self.rgb_bands = (55, 41, 12)
        elif dataset_name == 'Salinas':
            img = io.loadmat(folder + 'Salinas_corrected.mat')['salinas_corrected']
            gt = io.loadmat(folder + 'Salinas_gt.mat')['salinas_gt']
            self.label_values = ['Undefined', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow',
                                 'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained',
                                 'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
                                 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                                 'Vinyard_untrained', 'Vinyard_vertical_trellis']
            self.rgb_bands = (43, 21, 11)
        else:
            raise ValueError("{} dataset is unknown.".format(dataset_name))

        # Filter NaN values
        nan_mask = np.isnan(img.sum(axis=-1))
        if np.count_nonzero(nan_mask) > 0:
            print("Warning: NaN have been found in the data. "
                  "It is preferable to remove them beforehand. "
                  "Learning on NaN data is disabled.")
        img[nan_mask] = 0
        gt[nan_mask] = 0
        self.ground_truth = gt
        ignored_labels = [0]
        self.ignored_labels = list(set(ignored_labels))

        image = np.asarray(img, dtype='float32')
        image_height, image_width, image_bands = image.shape
        flat_image = np.reshape(image, (image_height * image_width, image_bands))
        self.image_bands = image_bands

        # Normalize data
        sca1 = StandardScaler()
        sca1.fit(flat_image)
        norm_img = sca1.transform(flat_image)  # Range [-1, 1]
        norm_img = (norm_img + 1) / 2  # Set image to the range [0, 1]
        self.image = np.reshape(norm_img, (image_height, image_width, image_bands))

    # Split ground-truth pixels into train, test, val
    def sample_dataset(self, train_size=0.8, val_size=0.1, max_train_samples=None):
        assert 1 >= train_size > 0, 'Train set size should be a value between 0 and 1'
        assert 1 > val_size >= 0, 'Validation set size should be a value between 0 and 1'
        assert train_size + val_size < 1, 'Train and validation sets should not use the whole dataset'

        # Get train samples and non-train samples (== test samples, when there is no validation set)
        train_gt, test_gt = self.split_ground_truth(self.ground_truth, train_size, max_train_samples)

        val_gt = None
        if val_size > 0:
            max_val_samples =\
                None if max_train_samples is None else int(max_train_samples * np.ceil(val_size / train_size))

            relative_val_size = val_size / (1 - train_size)
            val_gt, test_gt = self.split_ground_truth(test_gt, relative_val_size, max_val_samples)

        return train_gt, test_gt, val_gt

    @staticmethod
    def split_ground_truth(ground_truth, set1_size, max_samples=None):
        set1_gt = np.zeros_like(ground_truth)
        set2_gt = np.copy(ground_truth)

        set1_index_list = []
        for c in np.unique(ground_truth):
            if c == 0:
                continue
            class_indices = np.nonzero(ground_truth == c)
            index_tuples = list(zip(*class_indices))  # Tuples with (x, y) index values

            num_samples_set1 = int(np.ceil(set1_size * len(index_tuples)))
            set1_len = min(filter(lambda s: s is not None, [max_samples, num_samples_set1]))
            set1_index_list += random.sample(index_tuples, set1_len)

        set1_indices = tuple(zip(*set1_index_list))
        set1_gt[set1_indices] = ground_truth[set1_indices]
        set2_gt[set1_indices] = 0
        return set1_gt, set2_gt

    # Save information needed for testing
    def save_data(self, exec_folder):
        torch.save(self.image, exec_folder + 'proc_data.pth')

    # Load samples from hard drive for every run.
    @staticmethod
    def load_samples(split_folder, train_split, val_split, run):
        train_size = 'train_' + str(int(100 * train_split)) + '_'
        val_size = 'val_' + str(int(100 * val_split)) + '_'
        file = split_folder + train_size + val_size + 'run_' + str(run) + '.mat'
        data = io.loadmat(file)
        train_gt = data['train_gt']
        test_gt = data['test_gt']
        val_gt = data['val_gt']
        return train_gt, test_gt, val_gt

    # Save samples for every run.
    @staticmethod
    def save_samples(train_gt, test_gt, val_gt, split_folder, train_split, val_split, run):
        train_size = 'train_' + str(int(100 * train_split)) + '_'
        val_size = 'val_' + str(int(100 * val_split)) + '_'
        sample_file = split_folder + train_size + val_size + 'run_' + str(run) + '.mat'
        io.savemat(sample_file, {'train_gt': train_gt, 'test_gt': test_gt, 'val_gt': val_gt})


# Load a checkpoint
def load_checkpoint(checkpoint_folder, file):
    # Check whether to load latest checkpoint
    filename = checkpoint_folder + str(file)
    if file is None:
        file_type = '*.pth'
        files = glob.glob(checkpoint_folder + file_type)
        filename = max(files, key=os.path.getctime)

    # Load checkpoint
    loaded_checkpoint = torch.load(filename)

    # Load variable states
    first_run = loaded_checkpoint['run']
    first_epoch = loaded_checkpoint['epoch'] + 1
    loss_state = loaded_checkpoint['loss_state']
    correct_state = loaded_checkpoint['correct_state']
    values_state = (first_run, first_epoch, loss_state, correct_state)

    # Load dictionary states
    model_state = loaded_checkpoint['model_state']
    optimizer_state = loaded_checkpoint['optimizer_state']
    scheduler_state = loaded_checkpoint['scheduler_state']
    train_states = (model_state, optimizer_state, scheduler_state)

    # Load best models record
    best_models_dict = loaded_checkpoint['best_models_dict']
    return values_state, train_states, best_models_dict


def save_results(filename, report, run, epoch=-1, validation=False):
    mode = 'VALIDATION' if validation else 'TEST'

    epoch_str = ''
    if validation:
        assert epoch >= 0, 'Epoch should be a positive integer value'
        epoch_str = f' EPOCH {epoch}'

    with open(filename, 'a') as file:
        file.write(f'{mode} RESULTS FOR RUN {run + 1}{epoch_str}\n')
        file.write(f'\n- CLASSIFY REPORT:\n{report["classify_report"]}')
        file.write(f'\n- CONFUSION MATRIX:\n{report["confusion_matrix"]}\n')
        file.write(f'\n- PER CLASS ACCURACY:\n{report["class_accuracy"]}\n')
        file.write(f'\n- OVERALL ACCURACY: {report["overall_accuracy"]:f}\n')
        file.write(f'\n- AVERAGE ACCURACY: {report["average_accuracy"]:f}\n')
        file.write(f'\n- KAPPA COEFFICIENT: {report["kappa"]:f}\n')
        file.write('\n')
        file.write('#' * 70)
        file.write('\n\n')


# Initialize networks weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0, 0.05)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
    else:
        pass


# Creates matrix of label relations based on two vectors
def get_label_relations(labels1, labels2):
    length1, length2 = len(labels1), len(labels2)
    relations = torch.zeros([length1, length2])
    for i, label1 in enumerate(labels1):
        for j, label2 in enumerate(labels2):
            if label1.item() == label2.item():
                relations[i, j] = 1
    return relations
