""" Pytorch dataset. """
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import ndimage
import os.path

from . import utils
from . import data


class DetectionDataset(Dataset):
    """ Dataset with input+labels needed to train a Mask R-CNN with a single anchor box.

    We define an anchor box around each voxel and assign it to the cell with which it has
    the best overlap/highest IOU (or to none if it does not overlap with any cell).
    Following Ren et al., 2017 (Sec. 3.1.2), an anchor is positive if 1) if its IOU with
    any ground truth box is higher than some threshold (default 0.25 here) or 2) it has
    the highest IOU with a ground truth box.

    Arguments:
        examples: List of example ids to fetch.
        transform: A callable. Transform operation: receives an example (volume, label,
            anchor_label, anchor_iou, anchor_bbox) and returns a quintuple with each
            element transformed accordingly. All elements are numpy arrays.
        enhance_volume: Boolean. Whether to contrast normalize and sharpen input volumes.
        anchor_size: A triplet. Size of the anchor box in depth x height x width.
        iou_threshold: Float. Threshold used to determine an anchor as positive.

    Returns:
        A (volume, label, anchor_label, anchor_bbox) tuple.
            volume: A FloatTensor (1 x d x h x w): green channel of the stack.
            label: An IntTensor (d x h x w): voxelwise cell ids. Non-cell/background
                voxels have value 0. Cells have consecutive positive integers.
            cell_bbox: An IntTensor (6 x num_cells). Absolute bbox values for each cell.
            anchor_bbox: A FloatTensor (6 x d x h x w): Translation and scale invariant
                bounding box coordinates (z, y, x, d, h, w) parametrized as in Ren et
                al., 2017 (Eq. 3). NaN if anchor box is not assigned to any cell.


    Note on our coordinate system:
        We assume voxel (i, j, k) holds the value for coordinate (i+0.5, j+0.5, k+0.5).
        For instance, value in voxel (0, 0, 0) corresponds to coordinate (0.5, 0.5, 0.5).
        To bound voxels 0-5 a box will start at coordinate 0 and finish at coordinate 6.
    """
    def __init__(self, examples, transform=None, enhance_volume=False,
                 anchor_size=(15, 9, 9), iou_threshold=0.25):
        print('Creating dataset with examples:', examples)

        # Get volumes
        volumes_rel = data.Stack.Volume & [{'example_id': id_} for id_ in examples]
        volumes = volumes_rel.fetch('volume', order_by='example_id')
        if enhance_volume: # local contrast normalization
            volumes = [utils.lcn(v, (3, 25, 25)) for v in volumes]
        self.volumes = [np.expand_dims(volume, 0) for volume in volumes] # add channel dimension

        # Get labels
        labels_rel = data.Stack.Label & [{'example_id': id_} for id_ in examples]
        labels = labels_rel.fetch('label', order_by='example_id')
        self.labels = [lbl.astype(np.int32) for lbl in labels]

        # Get cell bboxes
        self.cell_bboxes = []
        for label in self.labels:
            cell_bbox = np.zeros((6, label.max()), dtype=np.float32) # absolute bbboxes per cell
            for i, cell_slices in enumerate(ndimage.find_objects(label)):
                cell_bbox[:3, i] = [(sl.start + sl.stop) / 2 for sl in cell_slices] # z, y, x
                cell_bbox[3:, i] = [sl.stop - sl.start for sl in cell_slices] # d, h, w
            self.cell_bboxes.append(cell_bbox)

        # Read anchor bboxes/labels for RPN (compute if needed)
        self.anchor_bboxes = []
        for example_id, label, cell_bbox in zip(examples, self.labels, self.cell_bboxes):
            filename = '/tmp/ex_{}_as_{}-{}-{}_iou_{}.mmap'.format(example_id,
                                                                   *anchor_size,
                                                                   iou_threshold)
            if not os.path.isfile(filename):
                # For each anchor, compute label of closest cell and iou with it
                anchor_lbl = np.zeros(label.shape, dtype=np.int32)
                anchor_iou = np.zeros(label.shape, dtype=np.float32)
                for cell_id, bbox in enumerate(cell_bbox.T, start=1):
                    # Compute iou with all intersecting anchors
                    intersection, iou_slices = _boxcar3d(bbox, anchor_size, label.shape)
                    iou = intersection / (bbox[3:].prod() + np.prod(anchor_size)
                                          - intersection)

                    # Update anchors whose iou is higher than previous one
                    to_update = iou > anchor_iou[iou_slices]
                    anchor_lbl[iou_slices][to_update] = cell_id
                    anchor_iou[iou_slices][to_update] = iou[to_update]

                # Compute parametrized bbox coordinates
                anchor_bbox = cell_bbox[:, anchor_lbl - 1].copy() # 6 x d x h x w
                zyx = np.stack(np.meshgrid(*[np.arange(d) + 0.5 for d in label.shape],
                                           indexing='ij'))
                as_ = np.reshape(anchor_size, (3, 1, 1, 1)) # to ease broadcasting
                anchor_bbox[:3] = (anchor_bbox[:3] - zyx) / as_ # (x - xa) / wa
                anchor_bbox[3:] = np.log(anchor_bbox[3:] / as_) # log(w / wa)

                # Select positive anchors (iou > thresh or highest iou with some cell)
                positive_anchors = anchor_iou >= iou_threshold
                for cell_id, cell_slices in enumerate(ndimage.find_objects(label), start=1):
                    cell_mask = np.logical_and(label[cell_slices] == cell_id, # inside cell mask
                                               anchor_lbl[cell_slices] == cell_id) # has the right label
                    if np.any(cell_mask): # some small cells may not have highest IOU with any anchor
                        max_iou = np.max(anchor_iou[cell_slices][cell_mask])
                        new_positives = np.logical_and(anchor_iou[cell_slices] == max_iou,
                                                       cell_mask)
                        positive_anchors[cell_slices][new_positives] = True
                anchor_bbox[:, ~positive_anchors] = float('nan')

                # Save anchor bbox as memmap array
                mmap_bbox = np.memmap(filename, dtype=np.float32, mode='w+',
                                      shape=anchor_bbox.shape)
                mmap_bbox[:] = anchor_bbox
                mmap_bbox.flush()
            mmap_bbox = np.memmap(filename, dtype=np.float32, mode='r',
                                  shape=(6, *label.shape))

            self.anchor_bboxes.append(mmap_bbox)

        # Store transform
        self.transform = transform

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, index):
        example = (self.volumes[index], self.labels[index], self.cell_bboxes[index],
                   self.anchor_bboxes[index])

        if self.transform is not None:
            example = self.transform(example)

        return tuple(torch.as_tensor(x) for x in example)


def _boxcar3d(bbox, filter_size, max_dim):
    """ Apply a boxcar 3D filter over the bbox.

    Argument:
        bbox: A 6-dim vector. Bbox: z, y, x, d, h, w.
        filter_size: Sequence of ints. Size of the anchor in z, y, x.
        max_dim: Sequence of ints. Maximum valid coordinate z, y, x.

    Returns:
        convolved3d: 3-d array. Bbox convolved with a 3-d tensor of filter_size size.
        extended_slices: A triplet. Slices of convolved3d in the original coordinate
            system.
    """
    extended_slices = []
    convolved1ds = []
    for x, d, fs, max_x in zip(bbox[:3], bbox[3:], filter_size, max_dim): # in z, y and x
        # Convolve 1-d filter of size fs over a vector of ones of size d (mode 'full')
        ramp = np.minimum(np.arange(1, d + 2 * (fs // 2) + 1), min(d, fs))
        conv1d = np.minimum(ramp, ramp[::-1])

        # Deal with edges
        low_coord, high_coord = int(round(x - d/2)) - fs//2, int(round(x + d/2)) + fs//2
        conv1d = conv1d[max(0, -low_coord): len(conv1d) - max(0,  high_coord - max_x)]
        convolved1ds.append(conv1d)
        extended_slices.append(slice(max(0, low_coord), min(max_x, high_coord)))

    # Create 3d convolution (outer product all dimensions)
    convolved3d = np.einsum('i, j, k -> ijk', *convolved1ds)

    return convolved3d, tuple(extended_slices)