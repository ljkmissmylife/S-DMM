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
        test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

        # Load model
        encoder_file = cfg.exec_folder + 'sdmm_encoder_run_' + str(run) + '.pth'
        relation_file = cfg.exec_folder + 'sdmm_relation_run_' + str(run) + '.pth'
        encoder_model = CNNEncoder().load_state_dict(torch.load(encoder_file))
        relation_model = RelationNetwork().load_state_dict(torch.load(relation_file))
        encoder_model.eval()
        relation_model.eval()

        # Set model to device
        if device == 'cuda':
            encoder_model = nn.DataParallel(encoder_model)
            relation_model = nn.DataParallel(relation_model)
        encoder_model = encoder_model.to(device)
        relation_model = relation_model.to(device)

        # Test model from the current run
        report = test_model(encoder_model, relation_model, test_loader, writer)
        filename = cfg.results_folder + 'test.txt'
        save_results(filename, report, run)

    if cfg.use_tensorboard:
        writer.close()


# TODO: Adapt that function
# Function for performing the tests for a given model and data loader
def test_model(encoder_model, relation_model, loader, writer=None):
    labels_pr = []
    prediction_pr = []
    with torch.no_grad():
        total_predicted = np.array([], dtype=int)
        total_labels = np.array([], dtype=int)
        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            # for images, labels in loader:
            # Get input and compute model output
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # Get predicted outputs
            _, predicted = torch.max(outputs, 1)

            # Save total values for analysis
            total_predicted = np.append(total_predicted, predicted.cpu().numpy())
            total_labels = np.append(total_labels, labels.cpu().numpy())

        report = get_report(total_predicted, total_labels)
        print(f'- Overall accuracy: {report["overall_accuracy"]:f}')
        print(f'- Average accuracy: {report["average_accuracy"]:f}')
        print(f'- Kappa coefficient: {report["kappa"]:f}')

        if writer is not None:
            # Accuracy per class
            classes = range(9)
            for i in classes:
                labels_i = labels_pr == i
                prediction_i = prediction_pr[:, i]
                writer.add_pr_curve(str(i), labels_i, prediction_i, global_step=0)

    return report


def original_test():
    # datasets prepare
    ''' img: array 3D; gt: array 2D;'''
    img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)
    # Number of classes + unidefind label
    N_CLASSES = len(LABEL_VALUES) - 1
    # Number of bands
    N_BANDS = img.shape[-1]
    # run the experiment several times
    for run in range(N_RUNS):
        # Sample get from training spectra
        train_gt, test_gt = get_sample(DATASET, SAMPLE_SIZE, run)

        ## test for all pixels
        if PRE_ALL:
            test_gt = np.ones_like(test_gt)

        print("{} samples selected (over {})".format(np.count_nonzero(train_gt),
                                                     np.count_nonzero(gt)))
        print("Running an experiment with run {}/{}".format(run + 1, N_RUNS))
        # for test
        train_dataset = HyperX(img, train_gt, DATASET, PATCH_SIZE, False, False)
        train_loader = Torchdata.DataLoader(train_dataset,
                                            batch_size=N_CLASSES * SAMPLE_SIZE,
                                            shuffle=False)
        tr_data, tr_labels = train_loader.__iter__().next()
        tr_data = tr_data.cuda(GPU)

        test_dataset = HyperX(img, test_gt, DATASET, PATCH_SIZE, False, False)
        test_loader = Torchdata.DataLoader(test_dataset,
                                           batch_size=1,
                                           shuffle=False)
        # init neural networks

        feature_encoder = CNNEncoder(N_BANDS, FEATURE_DIM)
        relation_network = RelationNetwork(PATCH_SIZE, FEATURE_DIM)

        # load weight from train
        if CHECKPOINT_ENCODER is not None:
            encoder_file = CHECKPOINT_ENCODER + 'non_augmentation_sample{}_run{}.pth'.format(SAMPLE_SIZE, run)
            with torch.cuda.device(GPU):
                feature_encoder.load_state_dict(torch.load(encoder_file))
        else:
            raise ('No Chenkpoints for Encoder Net')
        if CHECKPOINT_RELATION is not None:
            relation_file = CHECKPOINT_RELATION + 'non_augmentation_sample{}_run{}.pth'.format(SAMPLE_SIZE, run)
            with torch.cuda.device(GPU):
                relation_network.load_state_dict(torch.load(relation_file))
        else:
            raise ('No Chenkpoints for Relation Net')

        feature_encoder.cuda(GPU)
        relation_network.cuda(GPU)

        print('Testing...')
        feature_encoder.eval()
        relation_network.eval()
        accuracy, total = 0., 0.
        # scores_all = np.zeros((len(test_loader),N_CLASSES))
        test_labels = np.zeros(len(test_loader))
        pre_labels = np.zeros(len(test_loader))
        pad_pre_gt = np.zeros_like(test_dataset.label)
        pad_test_indices = test_dataset.indices
        for batch_idx, (te_data, te_labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
            with torch.no_grad():
                te_data, te_labels = te_data.cuda(GPU), te_labels.cuda(GPU)
                tr_features = feature_encoder(tr_data)
                te_features = feature_encoder(te_data)
                tr_features_ext = tr_features.unsqueeze(0)
                te_features_ext = te_features.unsqueeze(0).repeat(N_CLASSES * SAMPLE_SIZE, 1, 1, 1, 1)
                te_features_ext = torch.transpose(te_features_ext, 0, 1)
                trte_pairs = torch.cat((tr_features_ext, te_features_ext), 2).view(-1, FEATURE_DIM * 2, PATCH_SIZE,
                                                                                   PATCH_SIZE)
                trte_relations = relation_network(trte_pairs).view(-1, SAMPLE_SIZE)
                # scores = torch.mean(trte_relations,dim=1)
                scores, _ = torch.max(trte_relations, dim=1)
                # scores_all[batch_idx,:] = scores
                _, output = torch.max(scores, dim=0)
                pre_labels[batch_idx] = output.item() + 1
                test_labels[batch_idx] = te_labels.item() + 1
                pad_pre_gt[pad_test_indices[batch_idx]] = output.item() + 1
                accuracy += output.item() == te_labels.item()
                total += 1
        rate = accuracy / total
        print('Accuracy:', rate)
        # save sores
        results = dict()
        results['OA'] = rate
        results['gt'] = gt
        results['test_gt'] = test_gt
        results['train_gt'] = train_gt
        p = PATCH_SIZE // 2
        wp, hp = pad_pre_gt.shape
        pre_gt = pad_pre_gt[p:wp - p, p:hp - p]
        results['pre_gt'] = np.asarray(pre_gt, dtype='uint8')
        if PRE_ALL:
            results['pre_all'] = np.asarray(pre_gt, dtype='uint8')
        results['test_labels'] = test_labels
        results['pre_labels'] = pre_labels
        # expand train_gt by superpxiel
        save_folder = DATASET
        save_result(results, save_folder, SAMPLE_SIZE, run)


# Compute OA, AA and kappa from the results
def get_report(y_pr, y_gt):
    classify_report = metrics.classification_report(y_gt, y_pr)
    confusion_matrix = metrics.confusion_matrix(y_gt, y_pr)
    class_accuracy = metrics.precision_score(y_gt, y_pr, average=None)
    overall_accuracy = metrics.accuracy_score(y_gt, y_pr)
    average_accuracy = np.mean(class_accuracy)
    kappa_coefficient = kappa(confusion_matrix, 5)

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
