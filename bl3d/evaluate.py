import datajoint as dj
import numpy as np
import bisect
import torch
from torch.utils import data

from . import train
from . import datasets
from . import transforms
from . import params


schema = dj.schema('ecobost_bl3d3', locals())

#TODO: Compute mean IOU


@schema
class MeanAP(dj.Computed):
    definition = """ # object detection metrics

    -> train.TrainedModel
    -> params.EvalParams
    -> params.EvalSet
    ---
    map:            float       # mean average precision over all acceptance IOUs (same as COCO's mAP)
    mf1:            float       # mean F1 over all acceptance IOUs
    ap_50:          float       # average precision at IOU = 0.5 (default in Pascal VOC)
    ap_75:          float       # average precision at IOU = 0.75 (more strict)
    f1_50:          float       # F-1 at acceptance IOU = 0.5
    f1_75:          float       # F-1 at acceptance IOU = 0.75
    """

    class PerIOU(dj.Part):
        definition = """ # some metrics computed at a single acceptance IOUs

        -> master
        iou:                float       # acceptance iou used
        ---
        ap_precisions:      blob        # precisions at recall (0, 0.1, ... 0.9, 1)
        ap:                 float       # average precision (area under PR curve)
        tps:                int         # true positives
        fps:                int         # false positives
        fns:                int         # false negatives
        accuracy:           float
        precision:          float
        recall:             float
        f1:                 float       # F-1 score
        """

    def make(self, key):
        """ Compute mean average precision and other detection metrics

        Pseudocode for mAP as computed in COCO (Everingham et al., 2010; Sec 4.2):
            for each class
                for each image
                    Predict k bounding boxes and k confidences
                    Order by decreasing confidence
                    for each bbox
                        for each acceptance_iou in [0.5, 0.55, 0.6, ..., 0.85, 0.9, 0.95]
                            Find the highest IOU ground truth box that has not been assigned yet
                            if highest iou > acceptance_iou
                                Save whether bbox is a match (and with whom it matches)
                        accum results over all acceptance ious
                    accum results over all bboxes
                accum results over all images

                Reorder by decreasing confidence
                for each acceptance_iou in [0.5, 0.55, 0.6, ..., 0.85, 0.9, 0.95]:
                    Compute precision and recall at each example
                    for r in 0, 0.1, 0.2, ..., 1:
                        find precision at r as max(prec) at recall >= r
                    average all 11 precisions -> average precision at detection_iou
                average all aps -> average precision
            average over all clases -> mean average precision
        """
        print('Evaluating', key)
        eval_params = (params.EvalParams & key).fetch1()

        # Get model
        net = train.TrainedModel.load_model(key)
        net.nms_iou = eval_params['nms_iou']
        net.cuda()
        net.eval()

        # Get dataset
        examples = (params.TrainingSplit & key).fetch1('{}_examples'.format(key['eval_set']))
        enhance_volume = (params.TrainingParams & key).fetch1('enhanced_input')
        dataset = datasets.DetectionDataset(examples, transforms.ContrastNorm(),
                                            enhance_volume, net.anchor_size)
        dataloader = data.DataLoader(dataset, num_workers=2, pin_memory=True)

        # Accumulate true positives, false positives and false negatives across examples
        acceptance_ious = np.arange(0.5, 1, 0.05)
        num_gt_instances = 0 # number of ground truth instances
        num_pred_instances = 0 # number of predicted masks
        confidences = [] # confidence per predicted mask
        tps = np.empty([len(acceptance_ious), 0], dtype=bool) # ious x masks, whether mask is a match
        for volume, label, _, _ in dataloader:
            # Compute num_proposals and num_masks to use
            num_cells = np.prod(volume.shape) * eval_params['cells_per_um']
            num_proposals = int(round(num_cells * eval_params['proposal_factor']))
            num_masks = int(round(num_cells * eval_params['mask_factor']))

            # Create instance segmentations
            with torch.no_grad():
                probs, bboxes, masks = net.forward_eval(volume.cuda(), num_proposals,
                                                        num_masks)
            probs = 1 / (1 + np.exp(-probs)) # sigmoid
            masks = [1 / (1 + np.exp(-m)) for m in masks]
            label = label[0].numpy()

            # Compute limits of each mask (used for slicing below)
            low_indices = np.round(bboxes[:, :3] - bboxes[:, 3:] / 2 + 1e-7).astype(int)
            high_indices = np.round(bboxes[:, :3] + bboxes[:, 3:] / 2 + 1e-7).astype(int)

            # Match each predicted mask to a ground truth mask
            mask_tps = np.zeros([len(acceptance_ious), len(probs)], dtype=bool)
            gt_tps = np.zeros([len(acceptance_ious), label.max()], dtype=bool)
            for _, i, mask, low, high in sorted(zip(probs, range(len(probs)), masks,
                                                    low_indices, high_indices),
                                                reverse=True):
                # Reshape mask to full size
                full_mask = np.zeros(label.shape, dtype=bool)
                full_slices = tuple(slice(max(l, 0), max(h, 0)) for l, h in zip(low, high))
                mask_slices = tuple(slice(np.clip(-l, 0, h - l), h - l - np.clip(h - s, 0, h - l))
                                    for l, h, s in zip(low, high, label.shape))
                full_mask[full_slices] = mask[mask_slices] > eval_params['mask_threshold']

                # Assign mask to highest overlap match in ground truth label
                matches = find_matches(label, full_mask)
                for iou, match in sorted(matches, reverse=True):
                    is_acceptable = iou > acceptance_ious
                    is_unassigned = ~np.logical_or(gt_tps[:, match - 1], mask_tps[:, i])
                    mask_tps[:, i] = np.logical_and(is_acceptable, is_unassigned)
                    gt_tps[:, match - 1] = np.logical_and(is_acceptable, is_unassigned)

            # Accumulate results
            num_gt_instances += label.max()
            num_pred_instances += len(probs)
            confidences.extend(probs)
            tps = np.concatenate([tps, mask_tps], axis=1)

        # Compute precision and recall at each prediction point (after sorting by confidence)
        tps = tps[:, np.argsort(confidences)[::-1]]
        precision = np.cumsum(tps, 1) / np.arange(1, num_pred_instances + 1)
        recall = np.cumsum(tps, 1) / num_gt_instances

        # Add precisions at recall 0 and 1
        recall = np.concatenate([np.zeros((10, 1)), recall, np.ones((10, 1))], axis=1)
        precision = np.concatenate([precision[:, :1], precision, np.zeros((10, 1))], axis=1)

        # Make sure precision increases monotonically (from right to left)
        precision = np.maximum.accumulate(precision[:, ::-1], 1)[:, ::-1]

        # Compute mAP (area under the precision recall curve)
        ap_precisions = [[p[bisect.bisect_left(r, i)]  for i in np.linspace(0, 1, 11)]
                         for p, r in zip(precision, recall)]
        aps = np.mean(ap_precisions, axis=1) # per acceptance iou
        mAP = np.mean(aps)

        # Compute other metrics
        tp, fp, tn, fn = (tps.sum(1), num_pred_instances - tps.sum(1), 0,
                          num_gt_instances - tps.sum(1))
        metrics = compute_metrics(tp, fp, tn, fn)

        # Insert
        self.insert1({**key, 'map': mAP, 'mf1': metrics[1].mean(), 'ap_50': aps[0],
                      'ap_75': aps[5], 'f1_50': metrics[1][0], 'f1_75': metrics[1][5]})
        for (iou, ap_precisions_, ap, tp_, fp_, fn_, _, f1, accuracy, _, _, precision,
             recall) in zip(acceptance_ious, ap_precisions, aps, tp, fp, fn, *metrics):
            self.PerIOU().insert1({**key, 'iou': iou, 'ap_precisions': ap_precisions_,
                                   'ap': ap, 'tps': tp_, 'fps': fp_, 'fns': fn_,
                                   'accuracy': accuracy, 'precision': precision,
                                   'recall': recall, 'f1': f1})


def find_matches(labels, prediction):
    """ Find all labels that intersect with a given predicted mask (and their IOUs).

    Arguments:
        labels: Array with zeros for background and positive integers for each ground
            truth object in the volume.
        prediction: Boolean array with ones for the predicted mask. Same shape as labels.

    Returns:
        List of (iou, label) pairs.
    """
    res = []
    for m in filter(lambda x: x != 0, np.unique(labels[prediction])):
        label = labels == m
        union = np.logical_or(label, prediction)
        intersection = np.logical_and(label[union], prediction[union]) # bit faster
        iou = np.count_nonzero(intersection) / np.count_nonzero(union)
        res.append((iou, m))

    return res


def compute_metrics(true_positive, false_positive, true_negative, false_negative):
    """ Computes a set of different metrics given the confusion matrix values.

    Arguments:
        true_positive: Number of true positive examples/pixels.
        false_positive: Number of false positive examples/pixels.
        true_negative: Number of true negative examples/pixels.
        false_negative: Number of false negative examples/pixels.

    Returns:
        A septuple with IOU, F-1 score, accuracy, sensitivity, specificity, precision and
            recall.
    """
    epsilon = 1e-7 # To avoid division by zero

    # Evaluation metrics
    accuracy = (true_positive + true_negative) / (true_positive + true_negative +
                                                  false_positive + false_negative + epsilon)
    sensitivity = true_positive / (true_positive + false_negative + epsilon)
    specificity = true_negative / (false_positive + true_negative + epsilon)
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = sensitivity
    iou = true_positive / (true_positive + false_positive + false_negative + epsilon)
    f1 = (2 * precision * recall) / (precision + recall + epsilon)

    return iou, f1, accuracy, sensitivity, specificity, precision, recall