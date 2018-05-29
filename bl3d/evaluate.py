import datajoint as dj
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.nn import functional as F

from bl3d import train
from bl3d import datasets
from bl3d import transforms
from bl3d import params


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
        net.cuda()
        net.eval()

        # Get dataset
        examples = (train.Split() & key).fetch1('{}_examples'.format(key['set']))
        enhance_input = (params.TrainingParams() & key).fetch1('enhanced_input')
        dataset = datasets.SegmentationDataset(examples, transforms.ContrastNorm(),
                                               enhance_input)
        dataloader = DataLoader(dataset, num_workers=2, pin_memory=True)

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
            with torch.no_grad():
                for image, label in dataloader:
                    # Compute prediction (heatmap of probabilities)
                    output = forward_on_big_input(net, image.cuda())
                    prediction = F.softmax(output, dim=1) # 1 x num_classes x depth x height x width

                    # Threshold prediction to create segmentation
                    segmentation = prediction[0, 1].cpu().numpy() > threshold

                    # Accumulate confusion matrix values
                    confusion_matrix += compute_confusion_matrix(segmentation,
                                                                 label[0].numpy())

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

        threshold_metrics = {**key, 'thresholds': thresholds, 'tps': tps, 'fps': fps,
                             'tns': tns, 'fns':fns, 'accuracies': accuracies,
                             'precisions': precisions, 'recalls': recalls,
                             'specificities': specificities, 'ious': ious, 'f1s': f1s}
        self.ThresholdSelection().insert1(threshold_metrics)


@schema
class DetectionMetrics(dj.Computed):
    definition = """ # object detection metrics
    -> train.TrainedModel
    -> Set
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
        from skimage import measure
        import itertools
        import bisect

        print('Evaluating', key)

        # Get model
        net = train.TrainedModel.load_model(key)
        net.cuda()
        net.eval()

        # Get dataset
        examples = (train.Split() & key).fetch1('{}_examples'.format(key['set']))
        enhance_input = (params.TrainingParams() & key).fetch1('enhanced_input')
        dataset = datasets.SegmentationDataset(examples, transforms.ContrastNorm(),
                                               enhance_input, binarize_labels=False)
        dataloader = DataLoader(dataset, num_workers=2, pin_memory=True)

        # Accumulate true positives, false positives and false negatives over images
        acceptance_ious = np.arange(0.5, 1, 0.05)
        num_gt_instances = 0 # number of ground truth instances
        num_pred_instances = 0 # number of predicted masks
        confidences = [] # confidence per predicted mask
        tps = np.empty([len(acceptance_ious), 0], dtype=bool) # ious x masks, whether mask is a match
        for image, label in dataloader:
            # Create heatmap of predictions
            with torch.no_grad():
                output = forward_on_big_input(net, image.cuda())
                prediction = F.softmax(output, dim=1) # 1 x num_classes x depth x height x width

                pred = prediction[0, 1].cpu().numpy()
                label = label[0].numpy()

            # Create instance segmentation
            segmentation = _prob2labels(pred) # labels start at 1 and are sequential
            probs = [p.mean_intensity for p in measure.regionprops(segmentation, pred)]

            # Match each predicted mask
            mask_tps = np.zeros([len(acceptance_ious), segmentation.max()], dtype=bool)
            gt_tps = np.zeros([len(acceptance_ious), label.max()], dtype=bool)
            for _, mask_idx in sorted(zip(probs, itertools.count(1)), reverse=True):
                matches = find_matches(label, segmentation == mask_idx)
                for iou, match in sorted(matches, reverse=True):
                    is_acceptable = iou > acceptance_ious
                    is_unassigned = np.logical_and(~gt_tps[:, match - 1],
                                                   ~mask_tps[:, mask_idx - 1])
                    mask_tps[:, mask_idx - 1] = np.logical_and(is_acceptable, is_unassigned)
                    gt_tps[:, match - 1] = np.logical_and(is_acceptable, is_unassigned)

            # Accumulate results
            num_gt_instances += label.max()
            num_pred_instances += segmentation.max()
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


def forward_on_big_input(net, volume, max_size=256, padding=32, out_channels=2):
    """ Passes a big volume through a network dividing it in chunks.

    Arguments:
        net: A pytorch network.
        volume: The input to the network (num_examples x num_channels x d1 x d2 x ...).
        max_size: An int or tuple of ints. Maximum input size for every volume dimension.
        pad_amount: An int or tuple of ints. Amount of padding performed by the network.
            We discard an edge of this size out of chunks in the middle of the FOV to
            avoid padding effects. Better to overestimate.
        out_channels: Number of channels in the output.

    Note:
        Assumes net and volume are in the same device (usually both in GPU).
        If net is in train mode, each chunk will be batch normalized with diff parameters.
    """
    import itertools

    # Get some params
    spatial_dims = volume.dim() - 2 # number of dimensions after batch and channel

    # Basic checks
    listify = lambda x: [x] * spatial_dims if np.isscalar(x) else list(x)
    padding = [int(round(x)) for x in listify(padding)]
    max_size = [int(round(x)) for x in listify(max_size)]
    if len(padding) != spatial_dims or len(max_size) != spatial_dims:
        msg = ('padding and max_size should be a single integer or a sequence of the '
               'same length as the number of spatial dimensions in the volume.')
        raise ValueError(msg)
    if np.any(2 * np.array(padding) >= np.array(max_size)):
        raise ValueError('Padding needs to be smaller than half max_size.')

    # Evaluate input chunk by chunk
    output = torch.zeros(volume.shape[0], out_channels, *volume.shape[2:])
    for initial_coords in itertools.product(*[range(p, d, s - 2 * p) for p, d, s in
                                              zip(padding, volume.shape[2:], max_size)]):
        # Cut chunk (it starts at coord - padding)
        cut_slices = [slice(c - p, c - p + s) for c, p, s in zip(initial_coords, padding, max_size)]
        chunk = volume[(..., *cut_slices)]

        # Forward
        out = net(chunk)

        # Assign to output dropping padded amount (special treat for first chunk)
        output_slices = [slice(0 if sl.start == 0 else c, sl.stop) for c, sl in zip(initial_coords, cut_slices)]
        out_slices = [slice(0 if sl.start == 0 else p, None) for p, sl in zip(padding, cut_slices)]
        output[(..., *output_slices)] = out[(..., *out_slices)]

    return output


def compute_confusion_matrix(segmentation, label):
    """Confusion matrix for a single image: # of pixels in each category.

    Arguments:
        segmentation: Boolean array. Predicted segmentation.
        label: Boolean array. Expected segmentation.

    Returns:
        A quadruple with true positives, false positives, true negatives and false
            negatives
    """
    true_positive = np.sum(np.logical_and(segmentation, label))
    false_positive = np.sum(np.logical_and(segmentation, np.logical_not(label)))
    true_negative = np.sum(np.logical_and(np.logical_not(segmentation), np.logical_not(label)))
    false_negative = np.sum(np.logical_and(np.logical_not(segmentation), label))

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


def _prob2labels(pred):
    """ Transform voxelwise probabilities from a segmentation to instances.

    Pretty ad hoc. Bunch of numbers were manually chosen.

    Arguments:
        pred: Array with predicted probabilities.

    Returns:
        Array with same shape as pred with zero for background and positive integers for
            each predicted instance.
    """
    from skimage import filters, feature, morphology, measure, segmentation
    from scipy import ndimage

    # Find good binary threshold (may be a bit lower than best possible IOU, catches true cells that weren't labeled)
    thresh = filters.threshold_otsu(pred)

    # Find local maxima in the prediction heatmap
    smooth_pred = ndimage.gaussian_filter(pred, 0.7)
    peaks = feature.peak_local_max(smooth_pred, min_distance=4, threshold_abs=thresh,
                                   indices=False)
    markers = morphology.label(peaks)

    # Separate into instances based on distance
    thresholded = smooth_pred > thresh
    filled = morphology.remove_small_objects(morphology.remove_small_holes(thresholded), 65) # volume of sphere with diameter 5
    distance = ndimage.distance_transform_edt(filled)
    distance += 1e-7 * np.random.random(distance.shape) # to break ties
    label = morphology.watershed(-distance, markers, mask=filled)
    print(label.max(), 'initial cells')

    # Remove masks that are too small or too large
    label = morphology.remove_small_objects(label, 65)
    too_large = [p.label for p in measure.regionprops(label) if p.area > 4189]
    for label_id in too_large:
        label[label == label_id] = 0 # set to background
    label, _, _ = segmentation.relabel_sequential(label)
    print(label.max(), 'final cells')

    return label


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
    for m in np.delete(np.unique(labels[prediction]), 0):
        label = labels == m
        union = np.logical_or(label, prediction)
        intersection = np.logical_and(label[union], prediction[union]) # bit faster
        iou = np.count_nonzero(intersection) / np.count_nonzero(union)
        res.append((iou, m))

    return res