# -*- coding: utf-8 -*-
import datajoint as dj
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F

from bl3d import train
from bl3d import datasets
from bl3d import transforms


schema = dj.schema('ecobost_bl3d', locals())


@schema
class Set(dj.Lookup):
    definition = """ # set where metrics are computed
    set:                    varchar(8)
    """
    contents = [['train'], ['val']]


@schema
class SegmentationMetrics(dj.Computed):
    definition = """ # compute cross validation metrics per pixel
    -> train.TrainedModel
    -> Set
    ---
    best_threshold:         float
    best_iou:               float
    best_f1:                float
    """
    class ThresholdSelection(dj.Part):
        definition= """ # all thresholds tried
        -> master
        ---
        thresholds:         blob            # all thresholds tried
        tps:                blob            # true positives
        fps:                blob            # false positives
        tns:                blob            # true negatives
        fns:                blob            # false negatives
        accuracies:         blob            # accuracy at each threshold
        precisions:         blob            # precision at each threshold
        recalls:            blob            # recall/sensitivity at each threshold
        specificities:      blob            # specificity at each threshold
        ious:               blob            # iou at each threshold
        f1s:                blob            # F-1 score at each threshold
        """

    def make(self, key):
        print('Evaluating', key)

        # Get model
        net = train.TrainedModel.load_model(key)

        # Move model to GPU
        net.cuda()
        net.eval()

        # Get dataset
        examples = (train.Split() & key).fetch1('{}_examples'.format(key['set']))
        dataset = datasets.SegmentationDataset(examples, transforms.ContrastNorm())
        dataloader = DataLoader(dataset, num_workers=4)

        # Iterate over different probability thresholds
        thresholds = np.linspace(0, 1, 33)
        tps = []
        fps = []
        tns = []
        fns = []
        accuracies = []
        precisions = []
        recalls = []
        specificities = []
        ious = []
        f1s = []
        for threshold in thresholds:
            print('Threshold: {}'.format(threshold))

            confusion_matrix = np.zeros(4) # tp, fp, tn, fn
            for image, label in dataloader:

                with torch.no_grad():
                    # Compute prediction (heatmap of probabilities)
                    output = net(Variable(image.cuda()))
                    prediction = F.softmax(output, dim=1) # 1 x num_classes x depth x height x width

                # Threshold prediction to create segmentation
                segmentation = prediction[0, 1].data.cpu().numpy() > threshold

                # Accumulate confusion matrix values
                confusion_matrix += compute_confusion_matrix(segmentation, label.numpy())

            # Calculate metrics
            metrics = compute_metrics(*confusion_matrix)

            # Collect results
            tps.append(confusion_matrix[0])
            fps.append(confusion_matrix[1])
            tns.append(confusion_matrix[2])
            fns.append(confusion_matrix[3])
            accuracies.append(metrics[2])
            precisions.append(metrics[5])
            recalls.append(metrics[6])
            specificities.append(metrics[4])
            ious.append(metrics[0])
            f1s.append(metrics[1])

            print('IOU:', metrics[0])

        # Insert
        best_iou = max(ious)
        best_threshold = thresholds[ious.index(best_iou)]
        best_f1 = f1s[ious.index(best_iou)]
        self.insert1({**key, 'best_threshold': best_threshold, 'best_iou': best_iou,
                      'best_f1': best_f1})

        self.ThresholdSelection().insert1({**key, 'thresholds': thresholds, 'tps': tps,
                                           'fps': fps, 'tns': tns, 'fns':fns,
                                           'accuracies': accuracies, 'precisions': precisions,
                                           'recalls': recalls, 'specificities': specificities,
                                           'ious': ious, 'f1s': f1s})


def compute_confusion_matrix(segmentation, label):
    """Confusion matrix for a single image: # of pixels in each category."""
    true_positive = np.sum(np.logical_and(segmentation, label))
    false_positive = np.sum(np.logical_and(segmentation, np.logical_not(label)))
    true_negative = np.sum(np.logical_and(np.logical_not(segmentation), np.logical_not(label)))
    false_negative = np.sum(np.logical_and(np.logical_not(segmentation), label))

    return (true_positive, false_positive, true_negative, false_negative)


def compute_metrics(true_positive, false_positive, true_negative, false_negative):
    """ Computes a set of different metrics given the confusion matrix values."""
    epsilon = 1e-8 # To avoid division by zero

    # Evaluation metrics
    accuracy = (true_positive + true_negative) / (true_positive + true_negative +
                                                  false_positive + false_negative + epsilon)
    sensitivity = true_positive / (true_positive + false_negative + epsilon)
    specificity = true_negative / (false_positive + true_negative + epsilon)
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = sensitivity
    iou = true_positive / (true_positive + false_positive + false_negative + epsilon)
    f1 = (2 * precision * recall) / (precision + recall + epsilon)

    return (iou, f1, accuracy, sensitivity, specificity, precision, recall)

#TODO: class DetectionMetrics
    #best_mean_IOU per bounding box (by IOU I mean mean_IOU)
# use sensitivity and recall measures for detection too, (so per object rather than per pixel). One problem is how to deal with predictions that cover more than one ...
# true mask, I could compute IOu and assign it to that mask, (yet to solve if two predicted masks hit the same one are both right?). Should probably be a one to-one ...
# correspondence, each lab el mask should be related with the highest IOU predicted mask (any non-highest IOU will be considered as false positive)
# Other detecton metric, IOU oif bounding boxes


