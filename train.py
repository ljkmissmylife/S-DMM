#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 11:36 2021

@author: Pedro Vieira
"""

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from utils.tools import *
from utils.config import SDMMConfig
from utils.dataset import SDMMDataset
from net.encoder import CNNEncoder
from net.relation import RelationNetwork
from test import test_models

# Import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Trains the multiple runs with the whole dataset
def train():
    cfg = SDMMConfig('config.yaml')

    # Start tensorboard
    writer = None
    if cfg.use_tensorboard:
        writer = SummaryWriter(cfg.tensorboard_folder)

    # Load raw dataset, apply PCA and normalize dataset.
    data = HSIData(cfg.dataset, cfg.data_folder)

    # Load a checkpoint
    if cfg.use_checkpoint:
        print('Loading checkpoint')
        value_states, train_states, best_models_dict = load_checkpoint(cfg.checkpoint_folder,
                                                                       cfg.checkpoint_file)
        first_run, first_epoch, loss_state, correct_state = value_states
        model_state, optimizer_state, scheduler_state = train_states
        if first_epoch == cfg.num_epochs - 1:
            first_epoch = 0
            first_run += 1
        print(f'Loaded checkpoint with run {first_run} and epoch {first_epoch}')
    else:
        first_run, first_epoch, loss_state, correct_state = (0, 0, 0.0, 0)
        model_state, optimizer_state, scheduler_state = None, None, None
        best_models_dict = {'encoder': None, 'relation': None, 'accuracy': 0.0}

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

        # Create train and test dataset objects
        train_dataset = SDMMDataset(data.image, train_gt, cfg.sample_size, data_augmentation=True)
        val_dataset = SDMMDataset(data.image, val_gt, cfg.sample_size, data_augmentation=False)

        # Create train and test loaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=cfg.test_batch_size, shuffle=True)

        # Initialize neural networks
        encoder_model = CNNEncoder(data.image_bands, cfg.feature_dimensions)
        relation_model = RelationNetwork(cfg.sample_size, cfg.feature_dimensions)

        # Initialize weights
        encoder_model.apply(weights_init)
        relation_model.apply(weights_init)

        # Setup optimizer, loss and scheduler
        criterion = nn.MSELoss()

        encoder_opt = torch.optim.SGD(encoder_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        encoder_lrs = StepLR(encoder_opt, step_size=cfg.scheduler_step, gamma=cfg.gamma)

        relation_opt = torch.optim.SGD(relation_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        relation_lrs = StepLR(relation_opt, step_size=cfg.scheduler_step, gamma=cfg.gamma)

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
        criterion = criterion.to(device)
        encoder_model = encoder_model.to(device)
        relation_model = relation_model.to(device)
        total_steps = len(train_loader)
        for epoch in range(first_epoch, cfg.num_epochs):
            print("STARTING EPOCH {}/{}".format(epoch + 1, cfg.num_epochs))

            # Run iterations
            for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
                half_set = len(labels) // 2
                rest_set = len(labels) - half_set
                images1 = images[:half_set, :, :, :].to(device)
                images2 = images[half_set:, :, :, :].to(device)

                # Calculate features with encoder
                features1 = encoder_model(images1)
                features2 = encoder_model(images2)

                # Create matrix of feature maps for comparing all sample combinations
                features1_ext = features1.unsqueeze(1).repeat(1, rest_set, 1, 1, 1)
                features2_ext = features2.unsqueeze(0).repeat(half_set, 1, 1, 1, 1)

                # Concatenate pairs of samples and apply relation network
                relation_pairs = torch.cat((features1_ext, features2_ext), 2).view(-1, cfg.feature_dimensions * 2,
                                                                                   cfg.sample_size, cfg.sample_size)
                relations = relation_model(relation_pairs).view(-1, rest_set)

                label_relations = get_label_relations(labels[:half_set], labels[half_set:]).to(device)
                loss = criterion(relations, label_relations)

                # Backward and optimize
                encoder_opt.zero_grad()
                relation_opt.zero_grad()
                loss.backward()

                encoder_opt.step()
                relation_opt.step()

                encoder_lrs.step()
                relation_lrs.step()

                # Compute iteration accuracy and loss for later reporting
                running_loss += loss.item()
                predicted = (relations > cfg.test_threshold).int()
                num_correct = (predicted == label_relations).int().sum().item()
                running_correct += num_correct / (relations.shape[0] * relations.shape[1])

                # Print steps and loss every 'print_frequency'
                if (i + 1) % cfg.print_frequency == 0:
                    avg_loss = running_loss / cfg.print_frequency
                    accuracy = running_correct / cfg.print_frequency
                    tqdm.write(
                        f'\tEpoch [{epoch + 1}/{cfg.num_epochs}] Step [{i + 1}/{total_steps}]'
                        f'\tLoss: {avg_loss:.5f}\tAccuracy: {accuracy:.5f}')
                    running_loss = 0.0
                    running_correct = 0

                    # Compute intermediate results for visualization
                    if writer is not None:
                        # Write steps and loss every WRITE_FREQUENCY to tensorboard
                        writer.add_scalar('Training loss', avg_loss, epoch * total_steps + i)
                        writer.add_scalar('Accuracy', accuracy, epoch * total_steps + i)

            # Reset loss and correct predictions
            running_loss = 0.0
            running_correct = 0

            # Run validation
            if cfg.val_split > 0:
                print("STARTING VALIDATION {}/{}".format(epoch + 1, cfg.num_epochs))

                # Set models to eval mode
                encoder_model.eval()
                relation_model.eval()
                report = test_models(encoder_model, relation_model, val_loader, cfg.test_threshold)
                encoder_model.train()
                relation_model.train()

                # Save validation results
                filename = cfg.results_folder + 'validations.txt'
                save_results(filename, report, run, epoch, validation=True)

                if report['overall_accuracy'] > best_models_dict['accuracy']:
                    best_models_dict['encoder'] = encoder_model.state_dict()
                    best_models_dict['relation'] = relation_model.state_dict()
                    best_models_dict['accuracy'] = report['overall_accuracy']

            # Save checkpoint
            checkpoint = {
                'run': run,
                'epoch': epoch,
                'loss_state': running_loss,
                'correct_state': running_correct,
                'encoder_state': encoder_model.state_dict(),
                'relation_state': relation_model.state_dict(),
                'encoder_opt_state': encoder_opt.state_dict(),
                'relation_opt_state': relation_opt.state_dict(),
                'encoder_lrs_state': encoder_lrs.state_dict(),
                'relation_lrs_state': relation_lrs.state_dict(),
                'best_models_dict': best_models_dict
            }
            torch.save(checkpoint,
                       cfg.checkpoint_folder + 'checkpoint_run_' + str(run) + '_epoch_' + str(
                           epoch) + '.pth')

        # Reset first epoch in case a checkpoint was loaded
        first_epoch = 0

        # Save trained model
        # TODO: Save best models per run
        encoder_file = cfg.exec_folder + 'sdmm_encoder_run_' + str(run) + '.pth'
        relation_file = cfg.exec_folder + 'sdmm_relation_run_' + str(run) + '.pth'
        torch.save(encoder_model.state_dict(), encoder_file)
        torch.save(relation_model.state_dict(), relation_file)
        print(f'Finished training run {run + 1}')

    # Save the best model
    best_models_file = cfg.exec_folder + 'best_models.pth'
    torch.save(best_models_dict, best_models_file)

    if cfg.use_tensorboard:
        writer.close()


# Main function for running the train independently
def main():
    train()


if __name__ == '__main__':
    main()
