""" Some 3-D transforms specific to our dataset."""
import numpy as np


class RandomCrop:
    """ Randomly crop a 3-d patch from a volume.

    If a dimension is less or equal than desired size + exclude_border, crop will start
    at index zero.

    Arguments:
        size: int or triplet. Size of the crop in depth, height, width.
        exclude_border: int or triplet. Number of pixels to avoid near the edges.
    """
    def __init__(self, size=128, exclude_border=5):
        self.size = size if isinstance(size, tuple) else (size,) * 3
        self.exclude_border = (exclude_border if isinstance(exclude_border, tuple) else
                               (exclude_border,) * 3)

    def __call__(self, example):
        """ Apply transform.

        Arguments:
            example: (volume, label, cell_bbox, anchor_bbox) tuple: (c x d x h x w),
                (d x h x w), (6 x num_cells) and (6 x d x h x w) numpy arrays.
        """
        # Compute random slices of desired size
        max_indices = [max(0, dim - siz - exc) for dim, siz, exc in zip(example[1].shape,
                       self.size, self.exclude_border)]
        start = [np.random.randint(max_idx + 1) for max_idx in max_indices]
        slices = tuple(slice(st, st + siz) for st, siz in zip(start, self.size))

        # Crop
        cropped_volume = example[0][(..., *slices)]
        cropped_label = example[1][slices]
        cropped_abbox = example[3][(..., *slices)]

        # Relabel cells in cropped volume to be 1, ..., num_cells_in_cropped_label
        old_ids = np.delete(np.unique(cropped_label), 0)
        old_to_new = np.zeros(cropped_label.max() + 1, dtype=cropped_label.dtype)
        old_to_new[old_ids] = np.arange(len(old_ids)) + 1
        cropped_label = old_to_new[cropped_label]

        # Crop cell bboxes (drop bboxes outside volume and change coords)
        cropped_cbbox = example[2][:, old_ids - 1].copy()
        cropped_cbbox[:3] = cropped_cbbox[:3] - np.reshape(start, (3, 1))

        return (cropped_volume, cropped_label, cropped_cbbox, cropped_abbox)


class RandomRotate:
    """ Rotates an example by 0, 90, 180 or 270 degrees (at random) in the z axis."""
    def __call__(self, example):
        """ Apply transform.

        Arguments:
            example: (volume, label, cell_bbox, anchor_bbox) tuple: (c x d x h x w),
                (d x h x w), (6 x num_cells) and (6 x d x h x w) numpy arrays.
        """
        k = np.random.randint(4) # number of 90 degree rotations counterclockwise
        rotated_volume = np.rot90(example[0], k, axes=(-2, -1))
        rotated_label = np.rot90(example[1], k, axes=(-2, -1))
        rotated_cbbox = example[2].copy() # to avoid overwriting input
        rotated_abbox = np.rot90(example[3], k, axes=(-2, -1)).copy()

        # Fix cell coordinates (k?0: x, y; 1: y, w-x; 2: w-x, h-y, 3: h-y, x)
        h, w = example[1].shape[1:]
        rotated_cbbox[2] = (w - rotated_cbbox[2]) if k in [1, 2] else rotated_cbbox[2] # w - x
        rotated_cbbox[1] = (h - rotated_cbbox[1]) if k in [2, 3] else rotated_cbbox[1] # h - y
        if k in [1, 3]:
            rotated_cbbox[1:3] = rotated_cbbox[2:0:-1] # switch x <-> y
            rotated_cbbox[4:] = rotated_cbbox[:3:-1] # switch height <-> width

        # Fix anchor bbox coordinates (k?0: x, y; 1: y, -x; 2: -x, -y, 3: -y, x)
        rotated_abbox[2] = -rotated_abbox[2] if k in [1, 2] else rotated_abbox[2] # negate x
        rotated_abbox[1] = -rotated_abbox[1] if k in [2, 3] else rotated_abbox[1] # negate y
        if k in [1, 3]:
            rotated_abbox[1:3] = rotated_abbox[2:0:-1] # switch x <-> y
            rotated_abbox[4:] = rotated_abbox[:3:-1] # switch height <-> width

        return (rotated_volume, rotated_label, rotated_cbbox, rotated_abbox)


class RandomHorizontalFlip:
    """ Mirrors the example in the horizontal axis with prob=0.5.

    Note: Because we also rotate at 0, 90, 180 and 270, vertical flipping is redundant.
        For instance, rot0 + flip_v = rot180 + flip_h
    """
    def __call__(self, example):
        """ Apply transform.

        Arguments:
            example: (volume, label, cell_bbox, anchor_bbox) tuple: (c x d x h x w),
                (d x h x w), (6 x num_cells) and (6 x d x h x w) numpy arrays.
        """
        if np.random.random() > 0.5: # flip with equal probability
            flipped_volume = np.flip(example[0], axis=-1)
            flipped_label = np.flip(example[1], axis=-1)
            flipped_cbbox = example[2].copy() # to avoid overwriting input
            flipped_abbox = np.flip(example[3], axis=-1).copy()

            # Fix cell bbox coordinates (x -> w-x) and anchor bbox coordinates (x -> -x)
            flipped_cbbox[2] = example[1].shape[-1] - flipped_cbbox[2] # x -> w-x
            flipped_abbox[2] = -flipped_abbox[2] # x -> -x

            flipped = (flipped_volume, flipped_label, flipped_cbbox, flipped_abbox)
        else:
            flipped = example

        return flipped


class ContrastNorm:
    """ Normalizes the volume per channel to zero mean and stddev=1. Labels left
    untouched."""
    def __call__(self, example):
        """ Apply transform.

        Arguments:
            example: (volume, label, cell_bbox, anchor_bbox) tuple: (c x d x h x w),
                (d x h x w), (6 x num_cells) and (6 x d x h x w) numpy arrays.
        """
        norm = ((example[0] - np.mean(example[0], axis=(-3, -2, -1))) /
                np.std(example[0], axis=(-3, -2, -1)))

        return (norm, *example[1:])


class MakeContiguous:
    """ Makes example (numpy) arrays contiguous in memory.

    See: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    """
    def __call__(self, example):
        """ Apply transform.

        Arguments:
            example: (volume, label, cell_bbox, anchor_bbox) tuple: (c x d x h x w),
                (d x h x w), (6 x num_cells) and (6 x d x h x w) numpy arrays.
        """
        return tuple(np.ascontiguousarray(x) for x in example)