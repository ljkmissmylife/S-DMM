#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 11:51 2021

@author: Pedro Vieira
@description: Implements the test function for the S-DMM network
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
from tqdm import tqdm

from utils.config import SDMMConfig
from utils.dataset import SDMMDataset
from utils.tools import *
from net.encoder import CNNEncoder
from net.relation import RelationNetwork

# Import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################
# SET TEST CONFIG FILE #
########################
CONFIG_FILE = ''  # Empty string to load default 'config.yaml'


# Test S-DMM runs
def test():
    # Load config data from training
    config_file = 'config.yaml' if not CONFIG_FILE else CONFIG_FILE
    cfg = SDMMConfig(config_file, test=True)

    # Start tensorboard
    writer = None
    if cfg.use_tensorboard:
        writer = SummaryWriter(cfg.tensorboard_folder)

    # Load processed dataset
    data = torch.load(cfg.exec_folder + 'proc_data.pth')

    for run in range(cfg.num_runs):
        print(f'TESTING RUN {run + 1}/{cfg.num_runs}')

        # Load test ground truth and initialize test loader
        _, test_gt, _ = HSIData.load_samples(cfg.split_folder, cfg.train_split, cfg.val_split, run)
        test_dataset = SDMMDataset(data, test_gt, cfg.sample_size, data_augmentation=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=True)

        # Load model
        encoder_file = cfg.exec_folder + 'sdmm_encoder_run_' + str(run) + '.pth'
        relation_file = cfg.exec_folder + 'sdmm_relation_run_' + str(run) + '.pth'
        encoder_model = CNNEncoder()
        relation_model = RelationNetwork()
        encoder_model.load_state_dict(torch.load(encoder_file))
        relation_model.load_state_dict(torch.load(relation_file))
        encoder_model.eval()
        relation_model.eval()

        # Set model to device
        if device == 'cuda':
            encoder_model = nn.DataParallel(encoder_model)
            relation_model = nn.DataParallel(relation_model)
        encoder_model = encoder_model.to(device)
        relation_model = relation_model.to(device)

        # Test model from the current run
        report = test_models(encoder_model, relation_model, test_loader, cfg.test_threshold)
        filename = cfg.results_folder + 'test.txt'
        save_results(filename, report, run)

    if cfg.use_tensorboard:
        writer.close()


# Function for performing the tests for a given model and data loader
def test_models(encoder_model, relation_model, loader, threshold=0.5):
    with torch.no_grad():
        total_predicted = np.array([], dtype=int)
        total_labels = np.array([], dtype=int)
        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            # Divide input by two to create the comparison matrix
            half_set = len(labels) // 2
            rest_set = len(labels) - half_set
            images1 = images[:half_set, :, :, :].to(device)
            images2 = images[half_set:, :, :, :].to(device)

            # Calculate features with encoder
            features1 = encoder_model(images1)
            features2 = encoder_model(images2)

            feature_dimensions = features1.shape[1]
            sample_size = features1.shape[2]

            # Create matrix of feature maps for comparing all sample combinations
            features1_ext = features1.unsqueeze(1).repeat(1, rest_set, 1, 1, 1)
            features2_ext = features2.unsqueeze(0).repeat(half_set, 1, 1, 1, 1)

            # Concatenate pairs of samples and apply relation network
            relation_pairs = torch.cat((features1_ext, features2_ext), 2).view(-1, feature_dimensions * 2,
                                                                               sample_size, sample_size)
            relations = relation_model(relation_pairs).squeeze()
            predicted = (relations > threshold).int()
            label_relations = get_label_relations(labels[:half_set], labels[half_set:]).view(-1, 1).squeeze().int()

            # Save total values for analysis
            total_predicted = np.append(total_predicted, predicted.cpu().numpy())
            total_labels = np.append(total_labels, label_relations.numpy())

        report = get_report(total_predicted, total_labels)
        print(f'- Overall accuracy: {report["overall_accuracy"]:f}')
        print(f'- Average accuracy: {report["average_accuracy"]:f}')
        print(f'- Kappa coefficient: {report["kappa"]:f}')

    return report


# Compute OA, AA and kappa from the results
def get_report(y_pr, y_gt):
    classify_report = metrics.classification_report(y_gt, y_pr)
    confusion_matrix = metrics.confusion_matrix(y_gt, y_pr)
    class_accuracy = metrics.precision_score(y_gt, y_pr, average=None)
    overall_accuracy = metrics.accuracy_score(y_gt, y_pr)
    average_accuracy = np.mean(class_accuracy)
    kappa_coefficient = kappa(confusion_matrix, 2)

    # Save report values
    report = {
        'classify_report': classify_report,
        'confusion_matrix': confusion_matrix,
        'class_accuracy': class_accuracy,
        'overall_accuracy': overall_accuracy,
        'average_accuracy': average_accuracy,
        'kappa': kappa_coefficient
    }
    return report


# Compute kappa coefficient
def kappa(confusion_matrix, k):
    data_mat = np.mat(confusion_matrix)
    p_0 = 0.0
    for i in range(k):
        p_0 += data_mat[i, i] * 1.0
    x_sum = np.sum(data_mat, axis=1)
    y_sum = np.sum(data_mat, axis=0)
    p_e = float(y_sum * x_sum) / np.sum(data_mat)**2
    oa = float(p_0 / np.sum(data_mat) * 1.0)
    cohens_coefficient = float((oa - p_e) / (1 - p_e))
    return cohens_coefficient


# Main for running test independently
def main():
    test()


if __name__ == '__main__':
    main()
