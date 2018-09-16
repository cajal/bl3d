import datajoint as dj
import numpy as np
import bisect
import torch
from torch.utils import data

from . import train
from . import datasets
from . import transforms
from . import params
from . import utils


schema = dj.schema('ecobost_bl3d3', locals())


@schema
class SemanticMetrics(dj.Computed):
    definition = """ # standard voxel-wise segmentation metrics 
    
    -> train.QCANet
    -> params.EvalSet
    """

    @property
    def key_source(self):
        return super().key_source & {'eval_set': 'val'}

    class BestNDN(dj.Part):
        definition = """ # metrics for the bestndn model
        
        -> master
        threshold:              decimal(4, 3)   # (prob) threshold used for predictions   
        ---
        detection_iou:          float
        detection_f1:           float
        detection_accuracy:     float
        detection_precision:    float
        detection_recall:       float
        detection_specificity:  float 
        segmentation_iou:       float
        segmentation_f1:        float
        segmentation_accuracy:  float
        segmentation_precision: float
        segmentation_recall:    float
        segmentation_specificity:float 
        """

    class BestNSN(dj.Part):
        definition = """ # metrics for the bestnsn model
        
        -> master
        threshold:              decimal(4, 3)   # (prob) threshold used for predictions   
        ---
        detection_iou:          float
        detection_f1:           float
        detection_accuracy:     float
        detection_precision:    float
        detection_recall:       float
        detection_specificity:  float 
        segmentation_iou:       float
        segmentation_f1:        float
        segmentation_accuracy:  float
        segmentation_precision: float
        segmentation_recall:    float
        segmentation_specificity:float 
        """

    def make(self, key):
        print('Evaluating', key)

        # Get dataset
        examples_name = '{}_examples'.format(key['eval_set'])
        examples = (params.TrainingSplit & key).fetch1(examples_name)
        normalize_volume, centroid_radius = (params.TrainingParams & key).fetch1(
            'normalize_volume', 'centroid_radius')
        dataset = datasets.DetectionDataset(examples, transforms.ContrastNorm(),
                                            normalize_volume=normalize_volume,
                                            centroid_radius=centroid_radius)
        dataloader = data.DataLoader(dataset, num_workers=4, pin_memory=True)

        # Once for bestndn and once for bestnsn
        for model_name, model_rel in [('bestndn', self.BestNDN),
                                      ('bestnsn', self.BestNSN)]:
            # Get model
            net = train.QCANet.load_model(key, model_name)
            net.cuda()
            net.eval()

            # Iterate over images
            num_thresholds = 41
            thresholds = np.linspace(0, 1, num_thresholds)
            detection_tps = np.zeros(num_thresholds)  # true positives
            detection_fps = np.zeros(num_thresholds)  # false positives
            detection_tns = np.zeros(num_thresholds)  # true negatives
            detection_fns = np.zeros(num_thresholds)  # false negatives
            segmentation_tps = np.zeros(num_thresholds)
            segmentation_fps = np.zeros(num_thresholds)
            segmentation_tns = np.zeros(num_thresholds)
            segmentation_fns = np.zeros(num_thresholds)
            for volume, label, centroids in dataloader:
                # Get predictions
                with torch.no_grad():
                    detection, segmentation = net.forward_on_big_input(volume.cuda())
                    detection = torch.sigmoid(detection).squeeze().numpy()
                    segmentation = torch.sigmoid(segmentation).squeeze().numpy()

                # Compute voxel-wise confusion matrix
                for i, threshold in enumerate(thresholds):
                    tp, fp, tn, fn = compute_confusion_matrix(detection > threshold,
                                                              centroids[0].numpy())
                    detection_tps[i] += tp
                    detection_fps[i] += fp
                    detection_tns[i] += tn
                    detection_fns[i] += fn

                    tp, fp, tn, fn = compute_confusion_matrix(segmentation > threshold,
                                                              label[0].numpy())
                    segmentation_tps[i] += tp
                    segmentation_fps[i] += fp
                    segmentation_tns[i] += tn
                    segmentation_fns[i] += fn

                del volume, label, centroids, detection, segmentation

            # Compute metrics
            detection_metrics = compute_metrics(detection_tps, detection_fps,
                                                detection_tns, detection_fns)
            segmentation_metrics = compute_metrics(segmentation_tps, segmentation_fps,
                                                   segmentation_tns, segmentation_fns)

            # Insert
            self.insert1(key, skip_duplicates=True)
            for (threshold, detection_iou, detection_f1, detection_accuracy, _,
                 detection_specificity, detection_precision, detection_recall,
                 segmentation_iou, segmentation_f1, segmentation_accuracy, _,
                 segmentation_specificity, segmentation_precision, segmentation_recall) \
                    in zip(thresholds, *detection_metrics, *segmentation_metrics):
                model_rel.insert1({**key, 'threshold': threshold,
                                   'detection_iou': detection_iou,
                                   'detection_f1': detection_f1,
                                   'detection_accuracy': detection_accuracy,
                                   'detection_specificity': detection_specificity,
                                   'detection_precision': detection_precision,
                                   'detection_recall': detection_recall,
                                   'segmentation_iou': segmentation_iou,
                                   'segmentation_f1': segmentation_f1,
                                   'segmentation_accuracy': segmentation_accuracy,
                                   'segmentation_specificity': segmentation_specificity,
                                   'segmentation_precision': segmentation_precision,
                                   'segmentation_recall': segmentation_recall})


@schema
class InstanceMetrics(dj.Computed):
    definition = """  # instance segmentation metrics

    -> train.QCANet
    -> params.EvalSet
    """

    @property
    def key_source(self):
        return super().key_source & {'eval_set': 'val'}

    class BestNDNMuCov(dj.Part):
        definition = """ # average of the IOU of each predicted mask with its highest overlapping ground truth object 
        -> master
        threshold:              decimal(4, 3)   # (prob) Threshold used to generate the masks
        ---
        mucov:                  float
        """
        # MuCov = mean_i (max_j IOU(x_i, y_j)); where x_i is a predicted mask, and y_j is
        # a ground truth mask

    class BestNDNAP(dj.Part):
        definition=""" # metrics using a single acceptance IOU to determine correct detections
        -> master
        threshold:              decimal(4, 3)   # (prob) Threshold used to generate the masks
        iou:                    decimal(3, 2)
        ---
        precisions:             tinyblob        # precision at recall (0, 0.1, ... 0.9, 1)
        ap:                     float           # average precision across recalls (area under the PR curve)
        precision:              float           # proportion of predicted objects that are ground truth objects (IOU(x,y) > iou)  
        recall:                 float           # proportion of ground truth objects detected        
        f1:                     float           # F1-score
        """
        # # Pseudocode for mAP as computed in COCO (Everingham et al., 2010; Sec 4.2):
        # for each class
        #     for each image
        #         Predict k bounding boxes and k confidences
        #         Order by decreasing confidence
        #         for each bbox
        #             for each acceptance_iou in [0.5, 0.55, 0.6, ..., 0.85, 0.9, 0.95]
        #                 Find the highest IOU ground truth box that has not been assigned yet
        #                 if highest iou > acceptance_iou
        #                     Save whether bbox is a match (and with whom it matches)
        #                 accum results over all acceptance ious
        #             accum results over all bboxes
        #         accum results over all images
        #
        #     Reorder by decreasing confidence
        #     for each acceptance_iou in [0.5, 0.55, 0.6, ..., 0.85, 0.9, 0.95]:
        #         Compute precision and recall at each example
        #         for r in 0, 0.1, 0.2, ..., 1:
        #             find precision at r as max(prec) at recall >= r
        #         average all 11 precisions -> average precision at detection_iou
        #     average all aps -> average precision
        # average over all clases -> mean average precision

    class BestNSNMuCov(dj.Part):
        definition = """ # average of the IOU of each predicted mask with its highest overlapping ground truth object
        -> master
        threshold:              decimal(4, 3)   # (prob) Threshold used to generate the masks
        ---
        mucov:                  float
        """

    class BestNSNAP(dj.Part):
        definition = """ # metrics using a single acceptance IOU to determine correct detections
        -> master
        threshold:              decimal(4, 3)  # (prob) Threshold used to generate the masks
        iou:                    decimal(3, 2) 
        ---
        precisions:             tinyblob        # precision at recall (0, 0.1, ... 0.9, 1)
        ap:                     float           # average precision across recalls (area under the PR curve)
        precision:              float           # proportion of predicted objects that are ground truth objects (IOU(x,y) > iou)  
        recall:                 float           # proportion of ground truth objects detected        
        f1:                     float           # F1-score
        """

    def make(self, key):
        from skimage import measure

        print('Evaluating', key)

        # Get dataset
        examples_name = '{}_examples'.format(key['eval_set'])
        examples = (params.TrainingSplit & key).fetch1(examples_name)
        normalize_volume, centroid_radius = (params.TrainingParams & key).fetch1(
            'normalize_volume', 'centroid_radius')
        dataset = datasets.DetectionDataset(examples, transforms.ContrastNorm(),
                                            normalize_volume=normalize_volume,
                                            centroid_radius=centroid_radius,
                                            binarize_labels=False)
        dataloader = data.DataLoader(dataset, num_workers=4, pin_memory=True)

        # Once for bestndn and once for bestnsn
        for model_name, mucov_rel, ap_rel in [
            ('bestndn', self.BestNDNMuCov, self.BestNDNAP),
            ('bestnsn', self.BestNSNMuCov, self.BestNSNAP)]:

            # Get model
            net = train.QCANet.load_model(key, model_name)
            net.cuda()
            net.eval()

            # Set some parameters
            num_thresholds = 11
            thresholds = np.linspace(0.75, 0.95, num_thresholds)
            acceptance_ious = np.arange(0.5, 1, 0.05)
            num_ious = len(acceptance_ious)

            # Accumulate metrics across examples
            total_best_ious = np.zeros(num_thresholds)  # sum of max ious per prediction
            num_gt_instances = np.zeros(num_thresholds)  # number of ground truth instances
            num_pred_instances = np.zeros(num_thresholds)  # number of predicted masks
            confidences = [[] for _ in range(num_thresholds)]  # confidence per predicted mask
            tps = [np.empty([num_ious, 0], dtype=bool) for _ in range(num_thresholds)]  # ious x sorted_masks, whether mask is a match
            for volume, label, centroids in dataloader:
                # Get predictions
                with torch.no_grad():
                    detection, segmentation = net.forward_on_big_input(volume.cuda())
                    detection = torch.sigmoid(detection).squeeze().numpy()
                    segmentation = torch.sigmoid(segmentation).squeeze().numpy()
                    label = label[0].numpy()

                for i, threshold in enumerate(thresholds):
                    # Create instance segmentation
                    masks = utils.prob2labels(detection, segmentation, threshold)
                    mask_properties = measure.regionprops(masks, segmentation)
                    probs = [p.mean_intensity for p in mask_properties]

                    # Match each predicted mask to a ground truth mask
                    mask_tps = np.zeros([num_ious, len(probs)], dtype=bool)
                    gt_tps = np.zeros([num_ious, label.max()], dtype=bool)
                    mask_bboxes = np.array([p.bbox for p in mask_properties])
                    gt_bboxes = np.array([p.bbox for p in measure.regionprops(label)])
                    for mask_id, _ in sorted(enumerate(probs), key=lambda x: x[1],
                                             reverse=True):
                        # Find bbox containing mask and any overlapping gt object
                        mask_bbox = mask_bboxes[mask_id]
                        mask_slices = (slice(mask_bbox[0], mask_bbox[3]),
                                       slice(mask_bbox[1], mask_bbox[4]),
                                       slice(mask_bbox[2], mask_bbox[5]))
                        gt_ids = np.unique(label[mask_slices][masks[mask_slices] ==
                                                              (mask_id + 1)])
                        gt_indices = gt_ids[gt_ids != 0] - 1
                        overlapping_bboxes = gt_bboxes[gt_indices]

                        all_bboxes = np.concatenate([mask_bbox[None], overlapping_bboxes])
                        low_coords = np.min(all_bboxes[:, :3], axis=0)
                        high_coords = np.max(all_bboxes[:, 3:], axis=0)
                        full_slices = tuple(slice(l, h) for l, h in zip(low_coords,
                                                                        high_coords))

                        # Find all overlapping ground truth objects
                        matches = find_matches(label[full_slices],
                                               masks[full_slices] == (mask_id + 1))
                        sorted_matches = sorted(matches, reverse=True)

                        # Accumulate highest IOU for this predicted mask
                        if sorted_matches:
                            total_best_ious[i] += sorted_matches[0][0]

                        # Assign mask to highest overlap match in ground truth label
                        for iou, match in sorted_matches:
                            is_acceptable = iou > acceptance_ious
                            is_unassigned = ~np.logical_or(gt_tps[:, match - 1],
                                                           mask_tps[:, mask_id])
                            mask_tps[:, mask_id] = np.logical_and(is_acceptable,
                                                                  is_unassigned)
                            gt_tps[:, match - 1] = np.logical_and(is_acceptable,
                                                                  is_unassigned)

                    # Accumulate results
                    num_gt_instances[i] += label.max()
                    num_pred_instances[i] += len(probs)
                    confidences[i].extend(probs)
                    tps[i] = np.concatenate([tps[i], mask_tps], axis=1)

            # Compute MuCov
            mucov = total_best_ious / num_pred_instances

            # Compute precisions at (0, 0.1, ...0.9, 1.0) recall
            precisions = np.empty((num_thresholds, num_ious, 11))
            for i, (tps_, confidences_, num_pred_instances_, num_gt_instances_) in enumerate(
                    zip(tps, confidences, num_pred_instances, num_gt_instances)):
                # Compute precision and recall at each prediction point (after sorting by confidence)
                tps_ = tps_[:, np.argsort(confidences_)[::-1]]
                precision = np.cumsum(tps_, 1) / np.arange(1, num_pred_instances_ + 1)
                recall = np.cumsum(tps_, 1) / num_gt_instances_

                # Add precisions at recall 0 and 1
                recall = np.concatenate([np.zeros((num_ious, 1)), recall,
                                         np.ones((num_ious, 1))], axis=1)
                precision = np.concatenate([precision[:, :1], precision,
                                            np.zeros((num_ious, 1))], axis=1)

                # Make sure precision increases monotonically (from right to left)
                precision = np.maximum.accumulate(precision[:, ::-1], 1)[:, ::-1]

                # Compute mAP (area under the precision recall curve)
                precisions[i] = [[p[bisect.bisect_left(r, i)] for i in np.linspace(0, 1, 11)]
                                 for p, r in zip(precision, recall)]
            aps = np.mean(precisions, axis=-1)  # thresholds x ious

            # Compute other metrics
            tps = np.array([tps_.sum(1) for tps_ in tps])  # thresholds x ious
            tp, fp, tn, fn = (tps, num_pred_instances[:, None] - tps, 0,
                              num_gt_instances[:, None] - tps)
            _, f1, _, _, _, precision, recall = compute_metrics(tp, fp, tn, fn)

            # Insert
            self.insert1(key, skip_duplicates=True)
            for threshold, mucov_ in zip(thresholds, mucov):
                mucov_rel.insert1({**key, 'threshold': threshold, 'mucov': mucov_})
            for threshold_, row_precisions, row_aps, row_precision, row_recall, row_f1 in zip(
                    thresholds, precisions, aps, precision, recall, f1):
                for iou_, precisions_, ap_, precision_, recall_, f1_ in zip(
                        acceptance_ious, row_precisions, row_aps, row_precision,
                        row_recall, row_f1):
                    ap_rel.insert1({**key, 'threshold': threshold_, 'iou': iou_,
                                    'precisions': precisions_, 'ap': ap_,
                                    'precision': precision_, 'recall': recall_,
                                    'f1': f1_})


def compute_confusion_matrix(segmentation, label):
    """Confusion matrix for a single image: # of pixels in each category.

    Arguments:
        segmentation: Boolean array. Predicted segmentation.
        label: Boolean array. Expected segmentation.

    Returns:
        A quadruple with true positives, false positives, true negatives and false
            negatives
    """
    true_positive = np.logical_and(segmentation, label).sum()
    false_positive = np.logical_and(segmentation, np.logical_not(label)).sum()
    true_negative = np.logical_and(np.logical_not(segmentation), np.logical_not(label)).sum()
    false_negative = np.logical_and(np.logical_not(segmentation), label).sum()

    return (true_positive, false_positive, true_negative, false_negative)


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