#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:38:20 2018

@author: dengbin
"""

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from utils.tools import *
from utils.config import SDMMConfig
from utils.dataset import SDMMDataset
from net.encoder import CNNEncoder
from net.relation import RelationNetwork

# Import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters setting
DATASET = 'PaviaU'  # PaviaU; KSC; Salinas
SAMPLE_ALREADY = False  # whether randomly generated training samples are ready
N_RUNS = 2  # the runing times of the experiments
SAMPLE_SIZE = 10  # training samples per class
BATCH_SIZE_PER_CLASS = SAMPLE_SIZE // 2  # batch size of each class
PATCH_SIZE = 5  # Hyperparameter: patch size
FLIP_ARGUMENT = False  # whether need data argumentation of flipping data; default: False
ROTATED_ARGUMENT = False  # whether need data argumentation of rotated data; default: False
ITER_NUM = 1000  # the total number of training iter; default: 50000
TEST_NUM = 5  # the total number of test in the training process
SAMPLING_MODE = 'fixed_withone'  # fixed number for each class
FOLDER = './Datasets/'  # the dataset folder
LEARNING_RATE = 0.1  # 0.01 good / 0.1 fast for SGD; 0.001 for Adam
FEATURE_DIM = 64  # Hyperparameter: the number of convolutional filters


# Trains the multiple runs with the whole dataset
def train():
    cfg = SDMMConfig('config.yaml')

    # Start tensorboard
    writer = None
    if cfg.use_tensorboard:
        writer = SummaryWriter(cfg.tensorboard_folder)

    # Load raw dataset, apply PCA and normalize dataset.
    data = HSIData(cfg.dataset, cfg.data_folder, cfg.sample_bands)

    # Load a checkpoint
    if cfg.use_checkpoint:
        print('Loading checkpoint')
        value_states, train_states, best_model_state = load_checkpoint(cfg.checkpoint_folder,
                                                                       cfg.checkpoint_file)
        first_run, first_epoch, loss_state, correct_state = value_states
        model_state, optimizer_state, scheduler_state = train_states
        best_model, best_accuracy = best_model_state
        if first_epoch == cfg.num_epochs - 1:
            first_epoch = 0
            first_run += 1
        print(f'Loaded checkpoint with run {first_run} and epoch {first_epoch}')
    else:
        first_run, first_epoch, loss_state, correct_state = (0, 0, 0.0, 0)
        model_state, optimizer_state, scheduler_state = None, None, None
        best_model, best_accuracy = None, 0

        # Save data for tests if we are not loading a checkpoint
        data.save_data(cfg.exec_folder)

    # Run training
    print(f'Starting experiment with {cfg.num_runs} run' + ('s' if cfg.num_runs > 1 else ''))
    for run in range(cfg.num_runs):
        print(f'STARTING RUN {run + 1}/{cfg.num_runs}')

        # Generate samples or read existing samples
        if cfg.generate_samples and first_epoch == 0:
            train_gt, test_gt, val_gt = data.sample_dataset(cfg.train_split, cfg.val_split, cfg.max_samples)
            HSIData.save_samples(train_gt, test_gt, val_gt, cfg.split_folder, cfg.train_split, cfg.val_split, run)
        else:
            train_gt, _, val_gt = HSIData.load_samples(cfg.split_folder, cfg.train_split, cfg.val_split, run)

        # for test
        # train_dataset = HyperX(img, train_gt, DATASET, PATCH_SIZE, False, False)
        # train_loader = Torchdata.DataLoader(train_dataset,
        #                                     batch_size=N_CLASSES * SAMPLE_SIZE,
        #                                     shuffle=False)
        # tr_data, tr_labels = train_loader.__iter__().next()
        # if torch.cuda.is_available():
        #     tr_data, tr_labels = tr_data.cuda(GPU), tr_labels.cuda(GPU)

        # test_dataset = HyperX(img, test_gt, DATASET, PATCH_SIZE, False, False)
        # test_loader = Torchdata.DataLoader(test_dataset,
        #                                    batch_size=1,
        #                                    shuffle=False)

        # USED IN THE ACTUAL TRAINING
        # task_train_dataset.resetGt(task_train_gt)
        # task_train_loader = Torchdata.DataLoader(task_train_dataset,
        #                                          batch_size=N_CLASSES,
        #                                          shuffle=False)
        # task test
        # task_test_dataset.resetGt(task_test_gt)
        # task_test_loader = Torchdata.DataLoader(task_test_dataset,
        #                                         batch_size=N_CLASSES * BATCH_SIZE_PER_CLASS,
        #                                         shuffle=True)

        # Create train and test dataset objects
        train_dataset = SDMMDataset(data.image, train_gt, cfg.sample_size, data_augmentation=True)
        val_dataset = SDMMDataset(data.image, val_gt, cfg.sample_size, data_augmentation=False)

        # Create train and test loaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=cfg.test_batch_size, shuffle=False)

        # Initialize neural networks
        encoder_model = CNNEncoder(cfg.sample_bands, cfg.feature_dimensions)
        relation_model = RelationNetwork(cfg.sample_size, cfg.feature_dimensions)

        encoder_model.apply(weights_init)
        relation_model.apply(weights_init)

        # Setup optimizer, loss and scheduler
        criterion = nn.MSELoss()

        encoder_opt = torch.optim.SGD(encoder_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        encoder_lrs = StepLR(encoder_opt, step_size=cfg.scheduler_step, gamma=cfg.gamma)

        relation_opt = torch.optim.SGD(relation_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        relation_lrs = StepLR(relation_opt, step_size=cfg.scheduler_step, gamma=cfg.gamma)

        # training
        OA = np.zeros(TEST_NUM)
        oa_iter = 0
        test_iter = ITER_NUM // TEST_NUM
        display_iter = 10
        losses = np.zeros(ITER_NUM + 1)
        mean_losses = np.zeros(ITER_NUM + 1)

        # init torch data
        # task_train_dataset = HyperX(img, train_gt, DATASET, PATCH_SIZE, FLIP_ARGUMENT, ROTATED_ARGUMENT)
        # task_test_dataset = HyperX(img, train_gt, DATASET, PATCH_SIZE, FLIP_ARGUMENT, ROTATED_ARGUMENT)

        # Start counting loss and correct predictions
        running_loss = 0.0
        running_correct = 0

        # Load variable states when loading a checkpoint
        if cfg.use_checkpoint:
            # Encoder model
            encoder_model.load_state_dict(model_state[0])
            encoder_opt.load_state_dict(optimizer_state[0])
            encoder_lrs.load_state_dict(scheduler_state[0])

            # Relation model
            relation_model.load_state_dict(model_state[1])
            relation_opt.load_state_dict(optimizer_state[1])
            relation_lrs.load_state_dict(scheduler_state[1])

            running_loss = loss_state
            running_correct = correct_state

        # Run epochs
        encoder_model = encoder_model.to(device)
        relation_model = relation_model.to(device)
        total_steps = len(train_loader)
        for epoch in range(first_epoch, cfg.num_epochs):
            print("STARTING EPOCH {}/{}".format(epoch + 1, cfg.num_epochs))

            # Run iterations
            for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
                # for iter_ in tqdm(range(1, ITER_NUM + 1), desc='Training the network'):
                # images, labels = task_train_loader.__iter__().next()
                images = images.to(device)
                labels = labels.to(device)

                # Calculate features with encoder
                encoder_model.train()
                relation_model.train()

                sample_features = encoder_model(images)
                batch_features = encoder_model(batches)  # Why?

                # calculate relations
                # Really, why???
                # TODO: Figure this out
                sample_features_ext = sample_features.unsqueeze(0).repeat(N_CLASSES * BATCH_SIZE_PER_CLASS, 1, 1, 1, 1)
                batch_features_ext = batch_features.unsqueeze(0).repeat(N_CLASSES, 1, 1, 1, 1)
                batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
                relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, FEATURE_DIM * 2,
                                                                                              PATCH_SIZE, PATCH_SIZE)
                relations = relation_model(relation_pairs).view(-1, N_CLASSES)

                one_hot_labels = torch.zeros(N_CLASSES * BATCH_SIZE_PER_CLASS, N_CLASSES).scatter_(1,
                                                                                                   batch_labels.view(-1, 1),
                                                                                                   1)

                criterion = criterion.to(device)
                one_hot_labels = one_hot_labels.to(device)

                loss = criterion(relations, one_hot_labels)

                # Backward and optimize
                encoder_opt.zero_grad()
                relation_opt.zero_grad()
                loss.backward()

                encoder_opt.step()
                relation_opt.step()

                encoder_lrs.step()
                relation_lrs.step()

                # TODO: Continue here
                losses[i] = loss.item()
                mean_losses[i] = np.mean(losses[max(0, i - 10):i + 1])
                if i % cfg.print_frequency == 0:
                    string = 'Train (ITER_NUM {}/{})\tLoss: {:.6f}'
                    string = string.format(
                        i, ITER_NUM, mean_losses[i])
                    tqdm.write(string)

                # # Testing
                # if iter_ % test_iter == 0:
                #     print('Testing...')
                #     feature_encoder.eval()
                #     relation_network.eval()
                #     accuracy, total = 0., 0.
                #     for batch_idx, (te_data, te_labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
                #         with torch.no_grad():
                #             if torch.cuda.is_available():
                #                 te_data, te_labels = te_data.cuda(GPU), te_labels.cuda(GPU)
                #
                #             tr_features = feature_encoder(tr_data)
                #             te_features = feature_encoder(te_data)
                #             tr_features_ext = tr_features.unsqueeze(0)
                #             te_features_ext = te_features.unsqueeze(0).repeat(N_CLASSES * SAMPLE_SIZE, 1, 1, 1, 1)
                #             te_features_ext = torch.transpose(te_features_ext, 0, 1)
                #             trte_pairs = torch.cat((tr_features_ext, te_features_ext), 2).view(-1, FEATURE_DIM * 2,
                #                                                                                PATCH_SIZE, PATCH_SIZE)
                #             trte_relations = relation_network(trte_pairs).view(-1, SAMPLE_SIZE)
                #             # scores = torch.mean(trte_relations,dim=1)
                #             scores, _ = torch.max(trte_relations, dim=1)
                #             _, output = torch.max(scores, dim=0)
                #             accuracy += output.item() == te_labels.item()
                #             total += 1
                #     rate = accuracy / total
                #     OA[oa_iter] = rate
                #     oa_iter += 1
                #     print('Accuracy:', rate)
                #     # save networks
                #     save_encoder = 'Bing_Encoder'
                #     save_relation = 'Bing_Relation'
                #
                #     with ConditionalGpuContext(GPU):
                #         save_model(feature_encoder, save_encoder, train_loader.dataset.name, sample_size=SAMPLE_SIZE,
                #                    run=run, epoch=iter_, metric=rate)
                #         save_model(relation_network, save_relation, train_loader.dataset.name, sample_size=SAMPLE_SIZE,
                #                    run=run, epoch=iter_, metric=rate)
                #         if iter_ == ITER_NUM:
                #             model_encoder_dir = './checkpoints/' + save_encoder + '/' + train_loader.dataset.name + '/'
                #             model_relation_dir = './checkpoints/' + save_relation + '/' + train_loader.dataset.name + '/'
                #             model_encoder_file = model_encoder_dir + 'non_augmentation_sample{}_run{}.pth'.format(
                #                 SAMPLE_SIZE, run)
                #             model_relation_file = model_relation_dir + 'non_augmentation_sample{}_run{}.pth'.format(
                #                 SAMPLE_SIZE, run)
                #             torch.save(feature_encoder.state_dict(), model_encoder_file)
                #             torch.save(relation_network.state_dict(), model_relation_file)
            loss_dir = './results/losses/' + DATASET
            if not os.path.isdir(loss_dir):
                os.makedirs(loss_dir)
            loss_file = loss_dir + '/' + 'sample' + str(SAMPLE_SIZE) + '_run' + str(run) + '_dim' + str(
                FEATURE_DIM) + '.mat'
            io.savemat(loss_file, {'losses': losses, 'accuracy': OA})


# Main function for running the train independently
def main():
    train()


if __name__ == '__main__':
    main()
