""" Pytorch dataset. """
import torch
from torch.utils.data import Dataset
import numpy as np

class SegmentationDataset(Dataset):
    """ Dataset with green channel as volumes and binarized masks (nucleus/no nucleus) as
    labels.

    Arguments:
        examples:  List of example ids to fetch.
        transform: Transform operation (callable::(volume, label) -> (out_vol, out_lbl)).
            volume and label are numpy arrays.

    Returns:
        (volume, label) tuples. Volume is a 4-d FloatTensor (channels x depth x height x
            width) and label is a 3-d tensor (depth x height x width).
    """
    def __init__(self, examples, transform=None, enhanced=False):
        from bl3d import data

        print('Creating dataset with examples:', examples)

        # Get volumes
        if enhanced:
            volumes_rel = data.Stack.EnhancedVolume() & [{'example_id': id_} for id_ in examples]
        else:
            volumes_rel = data.Stack.Volume() & [{'example_id': id_} for id_ in examples]
        volumes = volumes_rel.fetch('volume', order_by='example_id')
        self.volumes = [np.expand_dims(volume, 0) for volume in volumes] # add channel dimension

        # Get labels
        labels_rel = data.Stack.Label() & [{'example_id': id_} for id_ in examples]
        labels = labels_rel.fetch('label', order_by='example_id')
        self.labels = [np.clip(masks, a_min=0, a_max=1).astype(int) for masks in labels] # binarize label

        # Store transform
        self.transform = transform

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, index):
        example = (self.volumes[index], self.labels[index])

        if self.transform is not None:
            example = self.transform(example)

        return tuple(x if torch.is_tensor(x) else torch.from_numpy(x) for x in example)