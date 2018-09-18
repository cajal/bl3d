import datajoint as dj
import numpy as np
import copy
import random
import time

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils import data
from torchvision.transforms import Compose
from torch.nn import functional as F

from . import params
from . import datasets
from . import transforms
from . import models


def log(*messages):
    """ Simple logging function."""
    formatted_time = "[{}]".format(time.ctime())
    print(formatted_time, *messages, flush=True)


schema = dj.schema('ecobost_bl3d3', locals())
dj.config['external-bl3d'] = {'protocol': 'file', 'location': '/mnt/scratch07/ecobost'}


@schema
class QCANet(dj.Computed):
    definition = """ # trained model and logs

    -> params.ModelParams                   # architectural details of the model to train
    -> params.TrainingParams                # hyperparameters used for training
    -> params.TrainingSplit                 # dataset split used for training
    ---
    train_ndn_loss:         longblob        # training loss of the ndn per batch 
    train_nsn_loss:         longblob        # training loss of the nsn per batch
    train_ndn_iou:          longblob        # iou of the ndn per training batch
    train_nsn_iou:          longblob        # iou of the nsn per training batch
    val_ndn_loss:           longblob        # validation loss of the ndn per epoch
    val_nsn_loss:           longblob        # validation loss of the nsn per epoch
    val_ndn_iou:            longblob        # iou of the ndn in the validation set
    val_nsn_iou:            longblob        # iou of the nsn in the validation set
    lr_history:             longblob        # learning rate per epoch
    bestndn_model:          external-bl3d   # dictionary with trained weights for the best ndn
    bestndn_ndn_iou:        float           # ndn iou when the ndn had the best validation iou
    bestndn_nsn_iou:        float           # nsn iou when the ndn had the best validation iou
    bestndn_loss:           float           # combined loss for the bestndn
    bestndn_epoch:          int             # epoch when the best ndn was found
    bestnsn_model:          external-bl3d   # dictionary with trained weights for the best nsn
    bestnsn_ndn_iou:        float           # ndn iou when the nsn had the best validation iou
    bestnsn_nsn_iou:        float           # nsn iou when the nsn had the best validation iou
    bestnsn_loss:           float           # combined loss for the bestnsn
    bestnsn_epoch:          int             # epoch when the best nsn was found
    training_time:          int             # how many minutes it took to train this network
    training_ts=CURRENT_TIMESTAMP:  timestamp   
    """
    def make(self, key):
        log('Training model', key['model_version'], 'with hyperparams',
            key['training_id'], 'using animal', key['val_animal'], 'for validation.')
        train_params = (params.TrainingParams & key).fetch1()

        # Set random seeds
        torch.manual_seed(train_params['seed'])
        torch.cuda.manual_seed(train_params['seed'])
        np.random.seed(train_params['seed'])
        random.seed(train_params['seed'])

        # Get datasets
        log('Creating datasets')
        dset_kwargs = {'normalize_volume': train_params['normalize_volume'],
                       'centroid_radius': train_params['centroid_radius']}

        train_examples = (params.TrainingSplit & key).fetch1('train_examples')
        train_transform = Compose([transforms.RandomCrop(train_params['train_crop_size']),
                                   transforms.RandomRotate(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ContrastNorm(),
                                   transforms.MakeContiguous()])
        train_dset = datasets.DetectionDataset(train_examples, train_transform,
                                               **dset_kwargs)
        train_dloader = data.DataLoader(train_dset, shuffle=True, num_workers=4,
                                        pin_memory=True)

        val_examples = (params.TrainingSplit & key).fetch1('val_examples')
        val_transform = Compose([transforms.RandomCrop(train_params['val_crop_size']),
                                 transforms.ContrastNorm()])
        val_dset = datasets.DetectionDataset(val_examples, val_transform, **dset_kwargs)
        val_dloader = data.DataLoader(val_dset, num_workers=4, pin_memory=True)

        # Get model
        log('Instantiating model')
        net = models.QCANet()
        net_version = (params.ModelParams & key).fetch1('core_version', 'ndn_version',
                                                        'nsn_version')
        if (net.core.version, net.ndn.version, net.nsn.version) != net_version:
            raise ValueError('Code and documented version do not match!')
        net.init_parameters()
        net.cuda()
        net.train()

        # Declare optimizer
        optimizer = optim.SGD(net.parameters(), lr=train_params['learning_rate'],
                              momentum=train_params['momentum'], nesterov=True,
                              weight_decay=train_params['weight_decay'])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   factor=train_params['lr_decay'],
                                                   patience=int(round(
                                                       train_params['decay_epochs'] /
                                                       train_params['val_epochs'])),
                                                   verbose=True)

        # Initialize some logs
        losses = {k: [] for k in ('ndn_train', 'nsn_train', 'ndn_val', 'nsn_val')}
        ious = {k: [] for k in ('ndn_train', 'nsn_train', 'ndn_val', 'nsn_val')}
        bestndn_ious = {k: 0 for k in ('ndn', 'nsn')}
        bestnsn_ious = {k: 0 for k in ('ndn', 'nsn')}
        best_epochs = {k: 0 for k in ('bestndn', 'bestnsn')}
        best_losses = {k: 0 for k in ('bestndn', 'bestnsn')}
        best_nets = {k: copy.deepcopy(net).cpu() for k in ('bestndn', 'bestnsn')}
        lr_history = []
        start_time = time.time()  # in seconds

        # Initialize some logs
        for epoch in range(1, train_params['num_epochs'] + 1):
            log('Epoch {}:'.format(epoch))

            # Record learning rate
            lr_history.append(optimizer.param_groups[0]['lr'])

            # Loop over training set
            for volume, label, centroids in train_dloader:
                # Zero the gradients
                net.zero_grad()

                # Move variables to GPU
                volume, label, centroids = volume.cuda(), label.cuda(), centroids.cuda()

                # Forward
                detection, segmentation = net(volume)

                # Compute loss
                ndn_loss = F.binary_cross_entropy_with_logits(detection[:, 0],
                                                              centroids.float(),
                                                              pos_weight=train_params['ndn_pos_weight'])
                nsn_loss = F.binary_cross_entropy_with_logits(segmentation[:, 0],
                                                              label.float(),
                                                              pos_weight=train_params['nsn_pos_weight'])
                loss = ndn_loss + train_params['nsn_loss_weight'] * nsn_loss

                # Check for divergence
                if (torch.isnan(ndn_loss) or torch.isinf(ndn_loss) or
                    torch.isnan(nsn_loss) or torch.isinf(nsn_loss)):
                    raise ValueError('Loss diverged')

                # Backprop
                loss.backward()
                optimizer.step()

                # Compute IOUs
                with torch.no_grad():
                    ndn_iou = _compute_iou(detection[0, 0], centroids,
                                           train_params['ndn_threshold'])
                    nsn_iou = _compute_iou(segmentation[0, 0], label,
                                           train_params['nsn_threshold'])

                # Record training losses
                losses['ndn_train'].append(ndn_loss.item())
                losses['nsn_train'].append(nsn_loss.item())
                ious['ndn_train'].append(ndn_iou.item())
                ious['nsn_train'].append(nsn_iou.item())
                log(('Training loss (iou * 100) for ndn / nsn: {:.5f} ({:04.2f}) / {:.5f}'
                     ' ({:04.2f})').format(ndn_loss.item(), ndn_iou.item() * 100,
                                           nsn_loss.item(), nsn_iou.item() * 100))

                # Delete variables to free memory (only minimal gain)
                del volume, label, centroids, detection, segmentation, ndn_loss, nsn_loss

            # Compute validation metrics and save best models
            if epoch % train_params['val_epochs'] == 0:

                # Compute loss and iou on the validation set
                total_ndn_loss = 0
                total_nsn_loss = 0
                total_ndn_iou = 0
                total_nsn_iou = 0
                with torch.no_grad():
                    net.eval()
                    for volume, label, centroids in val_dloader:
                        detection, segmentation = net(volume.cuda())

                        total_ndn_loss += F.binary_cross_entropy_with_logits(
                            detection[:, 0], centroids.float().cuda(),
                            pos_weight=train_params['ndn_pos_weight']).item()
                        total_nsn_loss += F.binary_cross_entropy_with_logits(
                            segmentation[:, 0], label.float().cuda(),
                            pos_weight=train_params['nsn_pos_weight']).item()
                        total_ndn_iou += _compute_iou(detection[0, 0], centroids.cuda(),
                                                      train_params['ndn_threshold']).item()
                        total_nsn_iou += _compute_iou(segmentation[0, 0], label.cuda(),
                                                      train_params['nsn_threshold']).item()

                        del volume, label, centroids, detection, segmentation
                    net.train()
                val_ndn_loss = total_ndn_loss / len(val_dloader)
                val_nsn_loss = total_nsn_loss / len(val_dloader)
                val_ndn_iou = total_ndn_iou / len(val_dloader)
                val_nsn_iou = total_nsn_iou / len(val_dloader)

                # Record validation loss
                losses['ndn_val'].append(val_ndn_loss)
                losses['nsn_val'].append(val_nsn_loss)
                ious['ndn_val'].append(val_ndn_iou)
                ious['nsn_val'].append(val_nsn_iou)
                log(('Validation loss (iou * 100) for ndn / nsn: {:.5f} ({:04.2f}) / '
                     '{:.5f} ({:04.2f})').format(val_ndn_loss, val_ndn_iou * 100,
                                                 val_nsn_loss, val_nsn_iou * 100))

                # Reduce learning rate
                scheduler.step(val_ndn_loss + train_params['nsn_loss_weight'] *
                               val_nsn_loss)

                # Save best model
                if val_ndn_iou > bestndn_ious['ndn']:
                    log('Saving bestndn model...')
                    bestndn_ious['ndn'] = val_ndn_iou
                    bestndn_ious['nsn'] = val_nsn_iou
                    best_losses['bestndn'] = val_ndn_loss + val_nsn_loss
                    best_epochs['bestndn'] = epoch
                    best_nets['bestndn'] = copy.deepcopy(net).cpu()
                if val_nsn_iou > bestnsn_ious['nsn']:
                    log('Saving bestnsn model')
                    bestnsn_ious['ndn'] = val_ndn_iou
                    bestnsn_ious['nsn'] = val_nsn_iou
                    best_losses['bestnsn'] = val_ndn_loss + val_nsn_loss
                    best_epochs['bestnsn'] = epoch
                    best_nets['bestnsn'] = copy.deepcopy(net).cpu()

        # Insert results
        results = key.copy()
        results['train_ndn_loss'] = np.array(losses['ndn_train'], dtype=np.float32)
        results['train_nsn_loss'] = np.array(losses['nsn_train'], dtype=np.float32)
        results['train_ndn_iou'] = np.array(ious['ndn_train'], dtype=np.float32)
        results['train_nsn_iou'] = np.array(ious['nsn_train'], dtype=np.float32)
        results['val_ndn_loss'] = np.array(losses['ndn_val'], dtype=np.float32)
        results['val_nsn_loss'] = np.array(losses['nsn_val'], dtype=np.float32)
        results['val_ndn_iou'] = np.array(ious['ndn_val'], dtype=np.float32)
        results['val_nsn_iou'] = np.array(ious['nsn_val'], dtype=np.float32)
        results['lr_history'] = np.array(lr_history, dtype=np.float32)
        results['bestndn_model'] = {k: v.cpu().numpy() for k, v in
                                    best_nets['bestndn'].state_dict().items()}
        results['bestndn_ndn_iou'] = bestndn_ious['ndn']
        results['bestndn_nsn_iou'] = bestndn_ious['nsn']
        results['bestndn_loss'] = best_losses['bestndn']
        results['bestndn_epoch'] = best_epochs['bestndn']
        results['bestnsn_model'] = {k: v.cpu().numpy() for k, v in
                                    best_nets['bestnsn'].state_dict().items()}
        results['bestnsn_ndn_iou'] = bestnsn_ious['ndn']
        results['bestnsn_nsn_iou'] = bestnsn_ious['nsn']
        results['bestnsn_loss'] = best_losses['bestnsn']
        results['bestnsn_epoch'] = best_epochs['bestnsn']
        results['training_time'] = round((time.time() - start_time) / 60)
        self.insert1(results)

    def load_model(key, bestndn_or_bestnsn='bestnsn'):
        # Declare network
        net = models.QCANet()
        net_version = (params.ModelParams & key).fetch1('core_version', 'ndn_version',
                                                        'nsn_version')
        if (net.core.version, net.ndn.version, net.nsn.version) != net_version:
            raise ValueError('Code and documented version do not match!')

        # Load state dict from database
        recarray = (QCANet & key).fetch1('{}_model'.format(bestndn_or_bestnsn))
        state_dict = {k: torch.as_tensor(np.array(recarray[k][0])) for k in
                      recarray.dtype.names}
        net.load_state_dict(state_dict)

        return net


def _compute_iou(pred, label, percentile_thresh):
    """ Compute iou given a prediction and label.

    Arguments:
        pred (torch.tensor): Prediction.
        label (torch.tensor): Label. Same shape as the prediction.
        percentile_thresh (float): Percentile of values in pred that will be considered
            positive (0-100).
    """
    binary_pred = pred > np.percentile(pred.detach().cpu().numpy(), percentile_thresh)
    iou = (binary_pred & label).sum().float() / (binary_pred | label).sum().float()
    return iou