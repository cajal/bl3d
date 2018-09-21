""" Pytorch dataset. """
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import ndimage

from . import utils
from . import data


class DetectionDataset(Dataset):
    """ Dataset with input+labels needed to train a segmentation and centroid detection
    network (a la Tokuoka et al., 2018).

    Arguments:
        examples (list): Example ids to fetch.
        transform (callable): Transform to apply to the example: receives a (volume,
            label, centroids) triplet and returns a triplet with the transformed example.
            All elements are numpy arrays.
        normalize_volume (bool): Whether to local contrast normalize the input.
        centroid_radius (int): Number of dilations to apply to a single point to produce a
            centroid mask. k will result in a 3-d disk of (2*k + 1) diameter.
        binarize_labels (bool): Whether labels would be binary or each cell will have a
            diff id.

    Returns:
        A (volume, label, centroids) tuple:
            volume (float32 array): A 1 x d x h x w tensor: green channel of the stack.
            label (uint8 or int32 array) A d x h x w array: voxelwise cell ids. Non-cell/
                background voxels have value 0. Cells have consecutive positive integers.
            centroids (uint8 array): A d x h x w array: centroids of all cells.
    """

    def __init__(self, examples, transform=None, normalize_volume=True, centroid_radius=2,
                 binarize_labels=True):
        print('Creating dataset with {}normalized examples {} and centroid radius {}'.format(
                '' if normalize_volume else 'un', examples, centroid_radius))

        # Get volumes
        volumes_rel = data.Stack.Volume & [{'example_id': id_} for id_ in examples]
        volumes = volumes_rel.fetch('volume', order_by='example_id')
        if normalize_volume:  # local contrast normalization
            volumes = [utils.lcn(v, (3, 25, 25)) for v in volumes]
        self.volumes = [np.expand_dims(volume, 0) for volume in volumes]  # add channel dimension

        # Get labels
        labels_rel = data.Stack.Label & [{'example_id': id_} for id_ in examples]
        labels = labels_rel.fetch('label', order_by='example_id')
        if binarize_labels:
            labels = [np.clip(lbl, a_min=0, a_max=1).astype(np.uint8) for lbl in labels]
        else:
            labels = [lbl.astype(np.int32) for lbl in labels]
        self.labels = labels

        # Get centroids
        centroids = []
        for example_id, volume in zip(examples, self.volumes):
            zs, ys, xs = (data.Stack.MaskProperties & {'example_id': example_id}).fetch(
                'z_centroid', 'y_centroid', 'x_centroid')
            zs, ys, xs = (np.round(zs).astype(int), np.round(ys).astype(int),
                          np.round(xs).astype(int))
            one_centroids = np.zeros(volume.shape[1:], dtype=np.bool)
            one_centroids[zs, ys, xs] = True  # initial centroids
            if centroid_radius > 0:
                one_centroids = ndimage.binary_dilation(one_centroids,
                                                        iterations=centroid_radius)
            centroids.append(one_centroids)
        self.centroids = [centroid.astype(np.uint8) for centroid in centroids]

        # Store transform
        self.transform = transform

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, index):
        example = (self.volumes[index], self.labels[index], self.centroids[index])

        if self.transform is not None:
            example = self.transform(example)

        return tuple(torch.as_tensor(x) for x in example)