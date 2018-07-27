""" Pytorch modules and utilities for Mask R-CNN (He et al., 2018). """
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def init_conv(modules):
    """ Initializes all module weights using He initialization and set biases to zero."""
    for module in modules:
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def init_bn(modules):
    """ Initializes all module weights to N(1, 0.1) and set biases to zero."""
    for module in modules:
        nn.init.normal_(module.weight, mean=1, std=0.1)
        nn.init.constant_(module.bias, 0)


class DenseNet(nn.Module):
    """ Computes intermediate feature representation of the image.

    A 5-layer dense block (Huang et al., 2018) with growth rate of 8 feature maps per
    layer. We use dilated convolutions to increase receptive field without downsampling
    FOV: dilation of 2 at layer 3 and 4, and 3 at layer 5.

    Input: N x 1 x D x H x W
    Output: N x 41 x D x H x W
    Effective receptive field: 17 x 17 x 17
    # params: 18 536
    """
    version = 1

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, 3, bias=False)
        self.bn1 = nn.BatchNorm3d(9)
        self.conv2 = nn.Conv3d(9, 8, 3, bias=False)
        self.bn2 = nn.BatchNorm3d(17)
        self.conv3 = nn.Conv3d(17, 8, 3, dilation=2, bias=False)
        self.bn3 = nn.BatchNorm3d(25)
        self.conv4 = nn.Conv3d(25, 8, 3, dilation=2, bias=False)
        self.bn4 = nn.BatchNorm3d(33)
        self.conv5 = nn.Conv3d(33, 8, 3, dilation=3)

    def forward(self, input_):
        padded = F.pad(input_, (9,) * 6, mode='replicate')
        a1 = torch.cat([padded[..., 1:-1, 1:-1, 1:-1], self.conv1(padded)], 1)
        a2 = torch.cat([a1[..., 1:-1, 1:-1, 1:-1], self.conv2(F.relu(self.bn1(a1), inplace=True))], 1)
        a3 = torch.cat([a2[..., 2:-2, 2:-2, 2:-2], self.conv3(F.relu(self.bn2(a2), inplace=True))], 1)
        a4 = torch.cat([a3[..., 2:-2, 2:-2, 2:-2], self.conv4(F.relu(self.bn3(a3), inplace=True))], 1)
        a5 = torch.cat([a4[..., 3:-3, 3:-3, 3:-3], self.conv5(F.relu(self.bn4(a4), inplace=True))], 1)
        return a5

    def init_parameters(self):
        init_conv([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5])
        init_bn([self.bn1, self.bn2, self.bn3, self.bn4])


class RPN(nn.Module):
    """ Region proposal network: predicts an objectness score and bounding box per voxel.

    A three-layer fully convolutional network: first filter is 3 x 3 (dilation 3), rest
    are 1 x 1. Feature maps: 48 -> 64 -> 7.

    Input: N x 41 x D x H x W
    Output: (N x D x H x W, N x 6 x D x H x W)
    Effective receptive field: 7 x 7 x 7
    # params: 56 887
    """
    version = 1

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(41, 48, 3, dilation=3, bias=False)
        self.bn1 = nn.BatchNorm3d(48)
        self.fc1 = nn.Conv3d(48, 64, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.fc2 = nn.Conv3d(64, 7, 1)

    def forward(self, input_):
        padded = F.pad(input_, (3,) * 6, mode='replicate')
        a1 = F.relu(self.bn1(self.conv1(padded)), inplace=True)
        a2 = F.relu(self.bn2(self.fc1(a1)), inplace=True)
        out = self.fc2(a2)
        return out[:, 0], out[:, 1:]

    def init_parameters(self):
        init_conv([self.conv1, self.fc1])
        init_bn([self.bn1, self.bn2])

        # Initialize last fuly connected layer to keep bbox predictions close to zero
        nn.init.kaiming_normal_(self.fc2.weight[:1])
        nn.init.normal_(self.fc2.weight[1:], mean=0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0)


class Bbox(nn.Module):
    """ Refine proposals made by the RPN.

    A standard convnet architecture:
        CONV(48) -> CONV(48)* -> CONV(48) -> AVGPOOL -> FC(96) -> FC(7)
    * This convolution has stride 2. It reduces the spatial dimensions of the volume by a
    factor of two.

    Input: N x 41 x D x H x W (D, H and W need to be even)
    Output: (N, N x 6)
    Effective receptive field: D x H x W
    # params: 183 319
    """
    version = 1

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(41, 48, 3, bias=False)
        self.bn1 = nn.BatchNorm3d(48)
        self.conv2 = nn.Conv3d(48, 48, 3, stride=2, bias=False) # 2x downsampling
        self.bn2 = nn.BatchNorm3d(48)
        self.conv3 = nn.Conv3d(48, 48, 3, bias=False)
        self.bn3 = nn.BatchNorm1d(48)
        self.fc1 = nn.Linear(48, 96, bias=False)
        self.bn4 = nn.BatchNorm1d(96)
        self.fc2 = nn.Linear(96, 7)

    def forward(self, input_):
        a1 = F.relu(self.bn1(self.conv1(input_)), inplace=True)
        a2 = F.relu(self.bn2(self.conv2(a1)), inplace=True)
        h3 = torch.mean(self.conv3(a2).view(*a2.shape[:2], -1), dim=-1) # global average pooling
        a3 = F.relu(self.bn3(h3), inplace=True)
        a4 = F.relu(self.bn4(self.fc1(a3)), inplace=True)
        out = self.fc2(a4)
        return out[:, 0], out[:, 1:]

    def init_parameters(self):
        init_conv([self.conv1, self.conv2, self.conv3, self.fc1])
        init_bn([self.bn1, self.bn2, self.bn3, self.bn4])

        # Initialize last fuly connected layer to keep initial bboxes close to anchors
        nn.init.kaiming_normal_(self.fc2.weight[:1])
        nn.init.normal_(self.fc2.weight[1:], mean=0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0)


class FCN(nn.Module):
    """ Segment each proposed region.

    A five-layer fully convolutional neural network:
        CONV(48) -> CONV(48)* -> CONV(48)* -> FC(96) -> FC(1)
    * This convolutions have dilation 2 to increase receptive field.

    Input: N x 41 x D x H x W
    Output: N x D x H x W
    Effective receptive field: 11 x 11 x 11
    # params: 182 737
    """
    version = 1

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(41, 48, 3, bias=False)
        self.bn1 = nn.BatchNorm3d(48)
        self.conv2 = nn.Conv3d(48, 48, 3, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm3d(48)
        self.conv3 = nn.Conv3d(48, 48, 3, dilation=2, bias=False)
        self.bn3 = nn.BatchNorm3d(48)
        self.fc1 = nn.Conv3d(48, 96, 1, bias=False)
        self.bn4 = nn.BatchNorm3d(96)
        self.fc2 = nn.Conv3d(96, 1, 1)

    def forward(self, input_):
        padded = F.pad(input_, (5,) * 6, mode='replicate')
        a1 = F.relu(self.bn1(self.conv1(padded)), inplace=True)
        a2 = F.relu(self.bn2(self.conv2(a1)), inplace=True)
        a3 = F.relu(self.bn3(self.conv3(a2)), inplace=True)
        a4 = F.relu(self.bn4(self.fc1(a3)), inplace=True)
        out = self.fc2(a4)
        return out[:, 0]

    def init_parameters(self):
        init_conv([self.conv1, self.conv2, self.conv3, self.fc1, self.fc2])
        init_bn([self.bn1, self.bn2, self.bn2, self.bn3, self.bn4])


class MaskRCNN(nn.Module):
    """ A Mask R-CNN (He et al., 2018). Performs instance segmentation.

    Image goes through an initial convolutional core that acts as a feature extractor, a
    region proposal network creates ROI proposals using this intermediate representation,
    non-maximum-suppression is performed to remove overlapping masks, top-k regions are
    classified and refined via a bbox prediction branch (CNN->MLP), NMS is performed
    again and the top-m regions are segmented by the mask branch.

    Arguments:
        anchor_size: Int or triplet. Size of the anchors used to create the labels.
        roi_size: Int or triplet. Size of the ROIs sent to the bbox and mask branch.
        nms_iou: Threshold for non-maximum suppression: if two bboxes have a an IOU
            higher than this, the one with lowest objectness score will be ignored.
        num_proposals: Number of proposals from the RPN that will go through the bbox
            refinement branch and mask branch. This changes during evaluation; see
            forward_eval().
    """
    def __init__(self, anchor_size=(15, 9, 9), roi_size=(12, 12, 12), nms_iou=0.25,
                 num_proposals=1024):
        super().__init__()

        self.core = DenseNet()
        self.rpn = RPN()
        self.bbox = Bbox()
        self.fcn = FCN()

        self.anchor_size = (anchor_size if isinstance(anchor_size, tuple) else
                            (anchor_size, ) * 3)
        self.roi_size = roi_size if isinstance(roi_size, tuple) else (roi_size, ) * 3
        self.nms_iou = nms_iou
        self.num_proposals = num_proposals # used only for training

    def forward(self, input_):
        """ Forward used during training.

        Arguments:
            input_: A 1 x 1 x D x H x W tensor. Input image. Works on single-image
                batches only.

        Returns:
            scores: A 1 x D x H x W tensor. Scores per proposal.
            proposals: A 1 x 6 x D x H x W tensor. Proposal bounding boxes.
            top_proposals: A NROIS x 6 array. Deparametrized bboxes of the proposals
                selected for further refinement.
            probs: A NROIS tensor. Probability score per each final bbox
            bboxes: A NROIS x 6 tensor. Bounding boxes of each final bbox.
            masks: A NROIS x R1 x R2 x R3 tensor. Heatmap of logits per bbox.
        """
        # Get intermediate representation
        hidden = self.core(input_)

        # Get top-k proposals (after NMS)
        scores, proposals = self.rpn(hidden)
        abs_proposals = deparametrize_rpn_proposals(proposals.detach().cpu().numpy(),
                                                    self.anchor_size)
        _, top_proposals = non_maximum_suppression(abs_proposals, scores.detach().cpu().numpy(),
                                                   self.nms_iou, stop_after=self.num_proposals)

        # ROI align
        roi_features = roi_align(hidden, top_proposals[0], self.roi_size) # NROIS x C x D x H x W

        # Refine bbox
        probs, bboxes = self.bbox(roi_features)

        # Segment
        masks = self.fcn(roi_features)

        return scores, proposals, top_proposals[0], probs, bboxes, masks

    def init_parameters(self):
        self.core.init_parameters()
        self.rpn.init_parameters()
        self.bbox.init_parameters()
        self.fcn.init_parameters()

    def forward_eval(self, input_, num_proposals, num_final_masks, block_size=160):
        """ Forward used during evaluation. This is non-differentiable.

        Information flow changes during evaluation: full size stacks will overflow memory
        so we apply the core and rpn per blocks and join the outputs; and the final
        segmentation is run in the refined bboxes (after a second and final round of NMS)
        rather than on the proposals.

        Arguments:
            input_: A 1 x 1 x D x H x W tensor. Input volume.
            num_proposals: Number of proposals to select after the RPN.
            num_final_masks: Number of refined bboxes that will go through the mask
                branch.
            block_size: An int. Size of the chunks to send through the core + rpn
                branches. Same size in all dimensions.

        Returns:
            top_probs: A NMASKS vector. Logits for each of the final detections.
            top_bboxes: A NMASKS x 6 array. Bboxes.
            top_masks: A list of (Di x Hi x Wi) arrays. Heatmap of logits for each final
                detection resampled to match each mask's size in the original image.
        """
        import itertools

        # Get intermediate representation and proposals (run in chunks)
        hidden = torch.empty(input_.shape[0], 41, *input_.shape[2:])
        scores = torch.empty(input_.shape[0], *input_.shape[2:])
        proposals = torch.empty(input_.shape[0], 6, *input_.shape[2:])

        padding = 12 # padding performed by the network, discarded of each output chunk
        for coords in itertools.product(*[range(0, d, block_size - 2 * padding) for d in
                                          input_.shape[2:]]):
            # Get next chunk
            cut_slices = [slice(c, c + block_size) for c in coords]
            chunk = input_[(..., *cut_slices)]

            # Forward
            chunk_hidden = self.core(chunk)
            chunk_scores, chunk_proposals = self.rpn(chunk_hidden)

            # Assign to output dropping padded amount (special treatment for first chunk)
            full_slices = [slice(0 if sl.start == 0 else sl.start + padding, sl.stop)
                             for sl in cut_slices]
            chunk_slices = [slice(0 if sl.start == 0 else padding, None) for sl in cut_slices]
            hidden[(..., *full_slices)] = chunk_hidden[(..., *chunk_slices)]
            scores[(..., *full_slices)] = chunk_scores[(..., *chunk_slices)]
            proposals[(..., *full_slices)] = chunk_proposals[(..., *chunk_slices)]

            del chunk_hidden, chunk_scores, chunk_proposals

        # Get top-k proposals (after NMS)
        abs_proposals = deparametrize_rpn_proposals(proposals.numpy(), self.anchor_size)
        _, top_proposals = non_maximum_suppression(abs_proposals, scores.numpy(),
                                                   self.nms_iou, stop_after=num_proposals)
        del scores, proposals

        #TODO: Run in batches over top_proposals (roi_align->bbbox->mask) if it overflows memory
#TODO: roi_align in CPU

        # ROI align
        roi_features = roi_align(hidden, top_proposals[0], self.roi_size)

        # Refine bbox
        probs, bboxes = self.bbox(roi_features) # NROIS, NROIS x 6
        del roi_features

        # Get top-k bboxes
        abs_bboxes = _deparametrize_bboxes(bboxes.detach().cpu().numpy(),
                                           zyx=top_proposals[0][:, :3],
                                           dhw=top_proposals[0][:, 3:])
        top_probs, top_bboxes = non_maximum_suppression(np.expand_dims(abs_bboxes.T, 0),
                                                        np.expand_dims(probs.detach().cpu().numpy(), 0),
                                                        self.nms_iou, stop_after=num_final_masks)
        bbox_features = roi_align(hidden, top_bboxes[0], self.roi_size)
        del probs, bboxes, hidden

        # Segment
        masks = self.fcn(bbox_features)
        top_masks = quantize_masks(masks.detach().cpu().numpy(), top_bboxes)

        return top_probs, top_bboxes, top_masks


def deparametrize_rpn_proposals(bboxes, anchor_size):
    """ Rewrite parametrized RPN proposals in the original coordinate system.

    Arguments:
        bboxes: Array (N x 6 x D x H x W) with bbox coordinates (z, y, x, d, h, w
            parametrized as in Ren et al., 2017) in the second dimension.
        anchor_size: Tripet or int. Size of the anchor (d, h, w) used in the
            parametrization.

    Returns:
        abs_bboxes: An array of the same shape as input with the absolute coordinates.
    """
    # Create 5-d anchor size and coordinates
    anchor_size = anchor_size if isinstance(anchor_size, tuple) else (anchor_size, ) * 3
    dhw = np.reshape(anchor_size, [1, 3, 1, 1, 1]) # all coords have the same d, h and w

    anchor_axes = [np.arange(0.5, s) for s in bboxes.shape[2:]]
    zyx = np.expand_dims(np.stack(np.meshgrid(*anchor_axes, indexing='ij')), 0) # 1 x 3 X D x H x W

    return _deparametrize_bboxes(bboxes, zyx, dhw)


def _deparametrize_bboxes(bboxes, zyx, dhw):
    """ Deparametrize bbox coordinates parametrized as in Ren et al., 2017.

    Arguments:
        bboxes: A (N x 6 x d1 x d2 x ...) array with bbox coordinates in the second
            dimension.
        zyx: An array ([1|N] x 3 x d1 x d2 x ...) with the z, y, x coordinates of the
            anchors used to parametrize the bboxes. Broadcastable with bboxes array.
        dhw: An array with the d, h, w dimensions of the anchors used to parametrize the
            bboxes. Has to be broadcastable with bboxes array.

    Returns:
        abs_bboxes: Array of same shape as input with absolute coordinates.
    """
    # Deparametrize bbox coordinates
    abs_bboxes = np.empty_like(bboxes)
    abs_bboxes[:, :3] = bboxes[:, :3] * dhw + zyx
    abs_bboxes[:, 3:] = np.exp(bboxes[:, 3:]) * dhw

    return abs_bboxes


def non_maximum_suppression(bboxes, scores, nms_iou=0.25, stop_after=5000,
                            block_size=512):
    """ Perform non maximum suppression for 3-D bounding boxes.

    Arguments:
        bboxes: Array (N x 6 x d1 x d2 x ...) with bbox coordinates (z, y, x, d, h, w).
        scores: Array (N x d1 x d2 x ...). Scores per bbox to use for NMS.
        nms_iou: Float. IOU used as threshold for suppression.
        stop_after: Int. Stop after finding this number of valid bounding boxes.
        block_size: Int. We try to add this number of bboxes in a loop.

    Returns:
        nms_scores: List of arrays; one per example. Scores of the selected bboxes.
        nms_bboxes: List of arrays; one per example (each is NROIS x 6). Selected bboxes.
    """
    nms_scores = []
    nms_bboxes = []
    for one_bboxes, one_scores in zip(bboxes, scores): # run nms per example
        # Reshape bboxes and scores
        one_bboxes = one_bboxes.reshape((6, -1)).T # N x 6
        one_scores = one_scores.ravel()

        # Sort scores, bboxes
        sort_order = np.argsort(one_scores)[::-1]
        selected_bboxes = np.empty((0, 6))
        selected_scores = np.empty(0)
        for ix in range(0, len(sort_order), block_size):
            # TODO: If using argpartition (and block_size + new possible ones is greater than stop after), order here
            next_bboxes = one_bboxes[sort_order[ix: ix + block_size]]
            next_scores = one_scores[sort_order[ix: ix + block_size]]

            # Discard bboxes that overlap highly with a previous bbox
            ious = compute_ious(next_bboxes, selected_bboxes) # block_size x num_selected
            good_bboxes = np.all(ious <= nms_iou, axis=-1)
            next_bboxes = next_bboxes[good_bboxes]
            next_scores = next_scores[good_bboxes]

            # Discard bboxes that overlap highly with someone else on the same set
            set_ious = compute_ious(next_bboxes, next_bboxes)
            # TODO: When using argpartition, sort and iterate over sorted indices or create a sorted black_ x black matrix. Better: sort bboxes when reading
            good_bboxes = np.ones(len(next_bboxes), dtype=bool)
            for i in range(len(next_bboxes)):
                if good_bboxes[i]:
                    good_bboxes[i + 1:][set_ious[i, i + 1:] > nms_iou] = False
            next_bboxes = next_bboxes[good_bboxes]
            next_scores = next_scores[good_bboxes]

            # Join to previously selected bboxes
            max_required = stop_after - len(selected_bboxes)
            selected_bboxes = np.concatenate([selected_bboxes, next_bboxes[:max_required]])
            selected_scores = np.concatenate([selected_scores, next_scores[:max_required]])

            # Stop if found enough bboxes
            if len(selected_bboxes) >= stop_after:
                break

        nms_scores.append(selected_scores)
        nms_bboxes.append(selected_bboxes)

    return nms_scores, nms_bboxes


def compute_ious(bboxes1, bboxes2):
    """ Compute iou between bboxes in bboxes1 and bboxes2.

    Arguments:
       bboxes1: N x 6 array. N bboxes (z, y, x, d, h, w).
       bboxes2: M x 6 array. M bboxes (z, y, x, d, h, w).

    Returns:
        ious: An array of size N x M with the corresponding ious
    """
    # Compute overlap in each dimension
    half_size1 = bboxes1[:, 3:] / 2
    half_size2 = bboxes2[:, 3:] / 2
    overlaps = []
    for i in range(3):
        first_coord = np.maximum.outer(bboxes1[:, i] - half_size1[:, i],
                                       bboxes2[:, i] - half_size2[:, i]) # N x M
        last_coord = np.minimum.outer(bboxes1[:, i] + half_size1[:, i],
                                      bboxes2[:, i] + half_size2[:, i])
        overlap1d = np.maximum(last_coord - first_coord, 0)
        overlaps.append(overlap1d)

    # Compute ious
    intersection = overlaps[0] * overlaps[1] * overlaps[2]
    union = np.add.outer(np.prod(bboxes1[:, 3:], axis=-1),
                         np.prod(bboxes2[:, 3:], axis=-1)) - intersection
    ious = intersection / union

    return ious


#def non_maximum_suppression2(bboxes, scores, nms_iou=0.25, stop_after=5000):
#    """ Perform non maximum suppression for 3-D bounding boxes.
#
#    Arguments:
#        bboxes: Array (N x 6 x d1 x d2 x ...) with bbox coordinates (z, y, x, d, h, w).
#        scores: Array (N x d1 x d2 x ...). Scores per bbox to use for NMS.
#        nms_iou: Float. IOU used as threshold for suppression.
#        stop_after: Int. Stop after finding this number of valid bounding boxes.
#
#    Returns:
#        nms_scores: List of arrays; one per example. Scores of the selected bboxes.
#        nms_bboxes: List of arrays; one per example (each is NROIS x 6). Selected bboxes.
#    """
#    nms_scores = []
#    nms_bboxes = []
#    for one_bboxes, one_scores in zip(bboxes, scores): # run nms per example
#        # Reshape bboxes and scores
#        one_bboxes = one_bboxes.reshape((6, -1)).T # N x 6
#        one_scores = one_scores.ravel()
#
#        # Save the highest nonoverlapping bboxes
#        top_indices = []
#        for next_index in np.argsort(one_scores)[::-1]:
#            ious = compute_ious2(one_bboxes[next_index], one_bboxes[top_indices])
#            if np.all(ious < nms_iou):
#                # Save it
#                top_indices.append(next_index)
#                if len(top_indices) >= stop_after:
#                    break
#
#        nms_scores.append(one_scores[top_indices])
#        nms_bboxes.append(one_bboxes[top_indices])
#
#    return nms_scores, nms_bboxes
#
#
#def compute_ious2(bbox, bboxes):
#    """ Compute iou of bbox with all bboxes.
#
#    Arguments:
#        bbox: Sixtuple: (z, y, x, d, h, w)
#        bboxes: N x 6 array. N bboxes.
#
#    Returns:
#        ious: An array of size N with the iou between bbox and every bbox in bboxes.
#    """
#    # Compute overlap in each dimension
#    first_coord = np.maximum(bbox[:3] - bbox[3:] / 2, bboxes[:, :3] - bboxes[:, 3:] / 2)
#    last_coord = np.minimum(bbox[:3] + bbox[3:] / 2, bboxes[:, :3] + bboxes[:, 3:] / 2)
#    overlap = np.maximum(last_coord - first_coord, 0) # when last_index was after first index
#
#    # Compute ious
#    intersection = np.prod(overlap, axis=-1)
#    union = np.prod(bboxes[:, 3:], axis=-1) + np.prod(bbox[3:]) - intersection
#    ious = intersection / union
#
#    return ious


def roi_align(features, bboxes, roi_size):
    """ ROI Align as described in He et al., 2018. This is a differentiable operation.

    Arguments:
        features. A tensor (1 x C x D x H x W): intermediate representation of the input.
        bboxes: An array (NROIS x 6). A sixtuple (x, y, z, d, h, w) of bbox coordinates.
        roi_size: A triplet (R1, R2, R3). The ROI size we want to extract.

    Returns:
        roi_features: A NROIS x C X R1 x R2 x R3 array. Extracted ROIs.

    Note: Points outside features will be filled with zeros.
    """
    roi_size = roi_size if isinstance(roi_size, tuple) else (roi_size, ) * 3
    roi_size = np.array(roi_size)

    # Find lowest/highest grid value in x, y, z for each bbox
    low_sample = bboxes[:, :3] - bboxes[:, 3:] / 2 + bboxes[:, 3:] / (4 * roi_size) # N x 3
    high_sample = bboxes[:, :3] + bboxes[:, 3:] / 2 - bboxes[:, 3:] / (4 * roi_size) # N x 3

    # Reparametrize to be between [-1, 1] (as needed for F.grid_sampler)
    volume_dhw = np.array(features.shape[-3:])
    low_gs = (2 * low_sample - volume_dhw) / (volume_dhw - 1)
    high_gs = (2 * high_sample - volume_dhw) / (volume_dhw - 1)

    # Sample ROI features
    roi_features = torch.empty(len(bboxes), features.shape[1], *roi_size,
                               device=features.device)
    for i, lp, hp in zip(range(len(bboxes)), low_gs, high_gs):
        # Create grid (numpy)
        coords = [np.linspace(l, h, 2 * rs) for l, h, rs in zip(lp, hp, roi_size)]
        grid = np.stack(np.meshgrid(*coords, indexing='ij'), axis=-1) # 2*R1 x 2*R2 x 2*R3 x 3

        # Sample the grid and pool (pytorch)
        g = torch.Tensor(np.expand_dims(grid, 0)).to(features.device)
        roi_features[i] = F.avg_pool3d(F.grid_sample(features, g), 2)

    return roi_features


def quantize_masks(masks, bboxes):
    """ Resample masks to match original quantization.

    We sample voxels at each .5 step. For instance:
        If the box goes from 3.2 to 4.7, the mask will be sampled at position 3.5 and 4.5
        If the box goes from 3.6 to 4.2, the mask will be an empty array (because no 0.5
        is intersected).

    Arguments:
        masks: An array (N x D x H x W). Masks.
        bboxes: An array (N x 6) with the bbox coordinates corresponding to the masks.

    Returns:
        res_masks: A list of 3-d arrrays. The resampled masks
    """
    from scipy import ndimage

    # Create sample coordinates (12.5, 13.5, ..., 21.5)
    low_sample = np.round(bboxes[:, :3] - bboxes[:, 3:] / 2 + 1e-7) + 0.5 # 0.5-1.5 -> 1.5, 1.5-2.5->2.5, ...
    high_sample = np.round(bboxes[:, :3] + bboxes[:, 3:] / 2 + 1e-7) - 0.5 # 0.5-1.5 -> 0.5, 1.5-2.5-> 1.5
    num_samples = np.round(high_sample - low_sample + 1).astype(int)

    # Convert sample coordinates into input to map_coordinates
    roi_size = np.array(masks.shape[1:])
    mpc_zero = bboxes[:, :3] - bboxes[:, 3:] / 2 + bboxes[:, 3:] / (2 * roi_size) # absolute coords equal to zero in map_coordinates
    low_mpc = (low_sample - mpc_zero) / (bboxes[:, 3:] / roi_size)
    high_mpc = (high_sample - mpc_zero) / (bboxes[:, 3:] / roi_size)

    # Resample each mask
    res_masks = []
    for mask, low, high, ns in zip(masks, low_mpc, high_mpc, num_samples):
        mpc_coords = [np.linspace(l, h, n) for l, h, n in zip(low, high, ns)]
        mpc_grid = np.stack(np.meshgrid(*mpc_coords, indexing='ij'))
        res_masks.append(ndimage.map_coordinates(mask, mpc_grid, mode='nearest'))

    return res_masks