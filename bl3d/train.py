""" Importing structural data from our pipeline. """
import datajoint as dj
import torch
import numpy as np
from torch import optim
from torch.optim import lr_scheduler
from torch.utils import data
from torchvision.transforms import Compose

from . import params
from . import datasets
from . import transforms
from . import models



def log(*messages):
    """ Simple logging function."""
    import time
    formatted_time = "[{}]".format(time.ctime())
    print(formatted_time, *messages, flush=True)


def mysql_float(number):
    """ Clean a float to be inserted in MySQL."""
    return -1 if np.isnan(number) else 1e10 if np.isinf(number) else number



schema = dj.schema('ecobost_bl3d2', locals())


@schema
class TrainedModel(dj.Computed):
    definition = """ # trained model and logs

    -> params.TrainingParams               # hyperparameters used for training
    -> params.ModelParams                  # architectural details of the model to train
    -> params.TrainingSplit                # dataset split used for training
    ---
    train_loss:             longblob        # training loss per example
    val_loss:               longblob        # validation loss per epoch
    diverged:               boolean         # whether the loss diverged during training (went to nan or inf)
    best_model:             longblob        # dictionary with trained weights
    best_epoch:             int             # epoch at which best_val_loss was achieved
    best_val_loss:          float           # best validation loss achieved
    best_train_loss:        float           # training loss averaged over all examples in best_epoch
    final_model:            longblob        # model in the final epoch
    final_val_loss:         float           # final loss in the validation set
    final_train_loss:       float           # training loss averaged over all examples in final_epoch
    training_ts=CURRENT_TIMESTAMP:  timestamp
    """
    def make(self, key):
        """ Trains a Mask R-CNN model using SGD with Nesterov's Accelerated Gradient."""
        log('Training model', key['model_version'], 'with hyperparams',
            key['training_id'], 'using animal', key['val_animal'], 'for validation.')
        train_params = (params.TrainingParams & key).fetch1()

        # Set random seeds
        torch.manual_seed(train_params['seed'])
        np.random.seed(train_params['seed'])

        # Get datasets
        log('Creating datasets')
        train_examples = (params.TrainingSplit & key).fetch1('train_examples')
        train_transform = Compose([transforms.RandomCrop(train_params['train_crop_size']),
                                   transforms.RandomRotate(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ContrastNorm(),
                                   transforms.MakeContiguous()])
        dset_kwargs = {'enhance_volume': train_params['enhanced_input'],
                       'anchor_size': tuple(train_params['anchor_size_' + d] for d in 'dhw')}
        train_dset = datasets.DetectionDataset(train_examples, train_transform, **dset_kwargs)
        train_dloader = data.DataLoader(train_dset, shuffle=True, num_workers=2, pin_memory=True)

        val_examples = (params.TrainingSplit & key).fetch1('val_examples')
        val_transform = Compose([transforms.RandomCrop(train_params['val_crop_size']),
                                 transforms.ContrastNorm()])
        val_dset = datasets.DetectionDataset(val_examples, val_transform, **dset_kwargs)
        val_dloader = data.DataLoader(val_dset, shuffle=True, num_workers=2, pin_memory=True)

        # Get model
        log('Instantiating model')
        net = models.MaskRCNN(anchor_size=dset_kwargs['anchor_size'],
                              roi_size=tuple(train_params['roi_size_' + d] for d in 'dhw'),
                              nms_iou=train_params['nms_iou'],
                              num_proposals=train_params['num_proposals'])
        if ((net.core.version, net.rpn.version, net.bbox.version, net.fcn.version) !=
            (params.ModelParams & key).fetch1('core_version', 'rpn_version',
                                              'bbox_version', 'fcn_version')):
            raise ValueError('Code and documented version do not match!')
        net.init_parameters()
        net.cuda()
        net.train()

        # Declare optimizer
        optimizer = optim.SGD(net.parameters(), lr=train_params['learning_rate'],
                              momentum=train_params['momentum'], nesterov=True,
                              weight_decay=train_params['weight_decay'])
        scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=train_params['lr_decay'],
                                             milestones=train_params['lr_schedule'])
        scheduler.step() # scheduler starts epochs at zero, we start them at 1

        # Initialize some logs
        train_loss = []
        val_loss = []
        best_model = net
        best_val_loss = 1e10 # float('inf')
        best_epoch = 0
        for epoch in range(1, train_params['num_epochs'] + 1):
            log('Epoch {}:'.format(epoch))

            # Loop over training set
            for volume, label, cell_bboxes, anchor_bboxes in train_dloader:
                # Zero the gradients
                net.zero_grad()

                # Forward
                scores, proposals, top_proposals, probs, bboxes, masks = net(volume.cuda())
                if len(top_proposals) < net.num_proposals:
                    print('Warning: Only', len(top_proposals), 'nonoverlapping proposals'
                          ' were generated.')

                # Create labels for the top proposals (passed through the bbox and mask branch)
                roi_bboxes, roi_masks = create_branch_labels(top_proposals, net.roi_size,
                                                             label[0].numpy(),
                                                             cell_bboxes[0].numpy().T)
                roi_bboxes = torch.cuda.FloatTensor(roi_bboxes)
                roi_masks = torch.cuda.ByteTensor(roi_masks.astype(np.uint8))

                # Compute loss
                anchor_bboxes = anchor_bboxes.cuda()
                loss = compute_loss(scores, ~torch.isnan(anchor_bboxes[:, 0]), proposals,
                                    anchor_bboxes, probs, ~torch.isnan(roi_bboxes[:, 0]),
                                    bboxes, roi_bboxes, masks, roi_masks,
                                    rpn_pos_weight=train_params['positive_weight'],
                                    smoothl1_weight=train_params['smoothl1_weight'])

                # Record training loss
                log('Training loss:', loss.item())
                train_loss.append(loss.item())

                # Check for divergence
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    log('Error: Loss diverged!')
                    del (volume, label, cell_bboxes, anchor_bboxes, scores, proposals,
                         top_proposals, probs, bboxes, masks, roi_bboxes, roi_masks,
                         loss) # free space

                    log('Inserting results')
                    results = key.copy()
                    results['train_loss'] = train_loss
                    results['val_loss'] = val_loss
                    results['diverged'] = True # !!
                    results['best_model'] = {k: v.cpu().numpy() for k, v in best_model.state_dict().items()}
                    results['best_epoch'] = best_epoch
                    results['best_val_loss'] = best_val_loss
                    best_train_loss = compute_loss_on_batch(best_model, train_dloader,
                                                            train_params['positive_weight'],
                                                            train_params['smoothl1_weight'])
                    results['best_train_loss'] = mysql_float(best_train_loss)
                    results['final_model'] = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
                    final_val_loss = compute_loss_on_batch(net, val_dloader,
                                                           train_params['positive_weight'],
                                                           train_params['smoothl1_weight'])
                    results['final_val_loss'] = mysql_float(final_val_loss)
                    final_train_loss = compute_loss_on_batch(net, train_dloader,
                                                             train_params['positive_weight'],
                                                             train_params['smoothl1_weight'])
                    results['final_train_loss'] = mysql_float(final_train_loss)
                    self.insert1(results)
                    return -1

                # Backprop
                loss.backward()
                optimizer.step()

                # Delete variables to free memory (only minimal gain)
                del (volume, label, cell_bboxes, anchor_bboxes, scores, proposals,
                     top_proposals, probs, bboxes, masks, roi_bboxes, roi_masks, loss)

            # Record validation loss
            epoch_val_loss = compute_loss_on_batch(net, val_dloader,
                                                   train_params['positive_weight'],
                                                   train_params['smoothl1_weight'])
            log('Validation loss:', epoch_val_loss)
            val_loss.append(epoch_val_loss)

            # Reduce learning rate
            scheduler.step()

            # Save best model
            if epoch_val_loss < best_val_loss:
                log('Saving best model...')
                best_val_loss = epoch_val_loss
                best_model = net
                best_epoch = epoch

        # Insert results
        log('Inserting results')
        results = key.copy()
        results['train_loss'] = train_loss
        results['val_loss'] = val_loss
        results['diverged'] = False
        results['best_model'] = {k: v.cpu().numpy() for k, v in best_model.state_dict().items()}
        results['best_epoch'] = best_epoch
        results['best_val_loss'] = best_val_loss
        results['best_train_loss'] = compute_loss_on_batch(best_model, train_dloader,
                                                           train_params['positive_weight'],
                                                           train_params['smoothl1_weight'])
        results['final_model'] = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
        results['final_val_loss'] = epoch_val_loss
        results['final_train_loss'] = compute_loss_on_batch(net, train_dloader,
                                                            train_params['positive_weight'],
                                                            train_params['smoothl1_weight'])
        self.insert1(results)

    def load_model(key, best_or_final='best'):
        # Declare network
        train_params = (params.TrainingParams & key).fetch1()
        net = models.MaskRCNN(anchor_size=tuple(train_params['anchor_size_' + d] for d in 'dhw'),
                              roi_size=tuple(train_params['roi_size_' + d] for d in 'dhw'),
                              nms_iou=train_params['nms_iou'],
                              num_proposals=train_params['num_proposals'])
        if ((net.core.version, net.rpn.version, net.bbox.version, net.fcn.version) !=
            (params.ModelParams & key).fetch1('core_version', 'rpn_version',
                                              'bbox_version', 'fcn_version')):
            raise ValueError('Code and documented version do not match!')

        # Load state dict from database
        recarray = (TrainedModel & key).fetch1('{}_model'.format(best_or_final))
        state_dict = {k: torch.from_numpy(recarray[k][0]) for k in recarray.dtype.names}
        net.load_state_dict(state_dict)

        return net


def create_branch_labels(bboxes, roi_size, label, gt_bboxes, iou_thresh=0.5):
    """ Create training labels for a set of bboxes.

    Arguments:
        bboxes: An array (N x 6). Bboxes for which labels will be created. Absolute bbox
            coordinates (z, y, x, d, h, w).
        roi_size: An int or triplet. Size of the roi to sample.
        label: An array (D x H x W) with the id of the object labelled at each voxel.
            Zero if no object is found in that voxel.
        gt_bboxes: An array (NOBJECTS x 6). Ground truth bboxes in absolute coords.
        iou_thresh: Float. Threshold to determine whether a bbox will be considered a
            positive object.

    Returns:
        par_bboxes: Parametrized bboxes (N x 6). Coordinates of the highest overlapping
            ground truth object parametrized (as in Ren et al., 2017) using the bbox as
            anchor. NaN for bboxes that do not overlap (IOU < iou_thresh) with any ground
            truth object.
        masks: A boolean array (N x R1 x R2 x R3) with the masks per bbox. All False for
            bboxes with no assigned object.
    """
    roi_size = roi_size if isinstance(roi_size, tuple) else (roi_size, ) * 3
    roi_size = np.array(roi_size)

    # Create parametrized bboxes and mask for each bbox
    par_bboxes = np.full_like(bboxes, np.nan)
    masks = np.zeros((len(bboxes), *roi_size), dtype=bool)
    for i, bbox in enumerate(bboxes):
        ious = models._compute_ious(bbox, gt_bboxes)
        if np.max(ious) >= iou_thresh:
            best_id = np.argmax(ious) + 1 # object_ids in label start at 1
            best_bbox = gt_bboxes[np.argmax(ious)]

            par_bboxes[i, :3] = (best_bbox[:3] - bbox[:3]) / bbox[3:]
            par_bboxes[i, 3:] = np.log(best_bbox[3:] / bbox[3:])

            coords = [np.linspace(x - d/2 + d/(2 * rs), x + d/2 - d/(2 * rs), rs) for
                      x, d, rs in zip(bbox[:3], bbox[3:], roi_size)] # roi coordinates
            indices = [np.round(c - 0.5 + 1e-7).astype(int) for c in coords] # nearest neighbor indices
            valid = [np.logical_and(idx >= 0, idx < max_idx) for idx, max_idx in
                     zip(indices, label.shape)]
            valid_label = label[np.ix_(*[idx[val] for idx, val in zip(indices, valid)])]
            masks[i][np.ix_(*valid)] = (valid_label == best_id)

    return par_bboxes, masks


def compute_loss(scores, scores_lbl, proposals, proposals_lbl, probs, probs_lbl, bboxes,
                 bboxes_lbl, masks, masks_lbl, rpn_pos_weight, smoothl1_weight):
    """ Compute Mask-RCNN (end-to-end) loss: L = L_{rpn} + L_{bbox} + L_{mask}.

    L_{rpn} and L_{bbox} (Ren et al. 2017; Eq. 3) are each the sum of the logistic loss
    on the scores and a smoothL1 loss (Girshick, 2015; Eq. 3) on the bbox parameters.
    L_{mask} is the average logistic loss across voxels.

    RCN and Mask-RCN sample proposals 1:3 and 1:1 (positive:negative), respectively, to
    compute the RPN loss function. Rather than discarding a lot of those proposals, we
    use a weighted loss function with rpn_pos_weight:1 weights.

    Arguments:
        scores: A N x D x H x W tensor. Logits (unnormalized log probs) per voxel.
        proposals: A N x 6 x D x H x W tensor. Parametrized bbox coordinates per voxel.
        probs: A NROIS tensor. Logits (unnormalized log probabilities) per ROI.
        bboxes: A NROIS x 6 tensor. Parametrized bbox coordinates per ROI.
        masks: A NROIS x R1 x R2 x R3 tensor. Logits (unnormalized log probabilities) per
            voxel.
        *_lbl: Tensors with the same shape as the predicted version. Labels.
        rpn_pos_weight: A float. Weight given to the predictions on positive RPN scores.
        smoothl1_weight: A float. Weight for the smooth L1 loss component in L_{rpn} and
            L_{bbox}.

    Returns:
        A scalar tensor/float. The total loss.

    Note:
        This is a differentiable function. All tensors must be in the same device.
    """
    from torch.nn import functional as F

    # Compute RPN loss
    weights = torch.ones_like(scores)
    weights[scores_lbl] = rpn_pos_weight
    rpn_class_loss = F.binary_cross_entropy_with_logits(scores, scores_lbl.float(),
                                                        weight=weights)
    rpn_bbox_loss = F.smooth_l1_loss(proposals.transpose(0, 1)[:, scores_lbl],
                                     proposals_lbl.transpose(0, 1)[:, scores_lbl])
    rpn_loss = rpn_class_loss + smoothl1_weight * rpn_bbox_loss

    # Compute bbox loss
    bbox_class_loss = F.binary_cross_entropy_with_logits(probs, probs_lbl.float())
    bbox_bbox_loss = F.smooth_l1_loss(bboxes[probs_lbl], bboxes_lbl[probs_lbl])
    bbox_loss = bbox_class_loss + smoothl1_weight * bbox_bbox_loss

    # Compute fcn loss
    fcn_loss = F.binary_cross_entropy_with_logits(masks[probs_lbl],
                                                  masks_lbl[probs_lbl].float())

    # Combine
    loss = rpn_loss + bbox_loss + fcn_loss

    return loss


def compute_loss_on_batch(net, dataloader, rpn_pos_weight, smoothl1_weight):
    """ Compute average loss over examples in a dataloader. """
    training_mode = net.training
    net.eval()

    total_loss = 0
    with torch.no_grad():
        for volume, label, cell_bboxes, anchor_bboxes in dataloader:
            # Forward
            scores, proposals, top_proposals, probs, bboxes, masks = net(volume.cuda())

            # Create labels for the top proposals (passed through bbox and mask branch)
            roi_bboxes, roi_masks = create_branch_labels(top_proposals, net.roi_size,
                                                         label[0].numpy(),
                                                         cell_bboxes[0].numpy().T)
            roi_bboxes = torch.cuda.FloatTensor(roi_bboxes)
            roi_masks = torch.cuda.ByteTensor(roi_masks)

            # Compute loss
            anchor_bboxes = anchor_bboxes.cuda()
            loss = compute_loss(scores, ~torch.isnan(anchor_bboxes[:, 0]), proposals,
                                anchor_bboxes, probs, ~torch.isnan(roi_bboxes[:, 0]),
                                bboxes, roi_bboxes, masks, roi_masks,
                                rpn_pos_weight=rpn_pos_weight,
                                smoothl1_weight=smoothl1_weight)
            total_loss += loss.item()
    loss = total_loss / len(dataloader)

    if training_mode:
        net.train()

    return loss