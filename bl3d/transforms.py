""" Some 3-D transforms specific to our dataset."""
import numpy as np


class RandomCrop:
    """ Randomly crop a 3-d patch from a volume.

    If a dimension is less or equal than desired size + exclude_border, crop will start
    at index zero.

    Arguments:
        size (triplet): Size of the crop in depth, height, width.
        exclude_border (triplet): Number of pixels to avoid near the edges.
    """

    def __init__(self, size=128, exclude_border=5):
        self.size = size if isinstance(size, tuple) else (size,) * 3
        self.exclude_border = (exclude_border if isinstance(exclude_border, tuple) else
                               (exclude_border,) * 3)

    def __call__(self, example):
        """ Apply transform.

        Arguments:
            example (triplet): Numpy arrays (c x d x h x w), (d x h x w) and (d x h x w).
        """
        # Compute random slices of desired size
        max_indices = [max(0, dim - siz - exc) for dim, siz, exc in zip(example[1].shape,
                                                                        self.size,
                                                                        self.exclude_border)]
        start = [np.random.randint(max_idx + 1) for max_idx in max_indices]
        slices = tuple(slice(st, st + siz) for st, siz in zip(start, self.size))

        return tuple(x[(..., *slices)] for x in example)


class RandomRotate:
    """ Rotates an example by 0, 90, 180 or 270 degrees (at random) in the z axis."""

    def __call__(self, example):
        """ Apply transform.

        Arguments:
            example (triplet): Numpy arrays (c x d x h x w), (d x h x w) and (d x h x w).
        """
        k = np.random.randint(4)  # number of 90 degree rotations counterclockwise
        return tuple(np.rot90(x, k, axes=(-2, -1)) for x in example)


class RandomHorizontalFlip:
    """ Mirrors the example in the horizontal axis with prob=0.5.

    Note:
        Because we also rotate at 0, 90, 180 and 270, vertical flipping is redundant. For
            instance, rot0 + flip_v = rot180 + flip_h
    """

    def __call__(self, example):
        """ Apply transform.

        Arguments:
            example (triplet): Numpy arrays (c x d x h x w), (d x h x w) and (d x h x w).
        """
        flip = np.random.random() > 0.5  # flip with equal probability
        return tuple(np.flip(x, axis=-1) for x in example) if flip else example


class ContrastNorm:
    """ Normalizes the volume per channel to zero mean and stddev=1. Labels left
    untouched."""

    def __call__(self, example):
        """ Apply transform.

        Arguments:
            example (triplet): Numpy arrays (c x d x h x w), (d x h x w) and (d x h x w).
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
            example (triplet): Numpy arrays (c x d x h x w), (d x h x w) and (d x h x w).
        """
        return tuple(np.ascontiguousarray(x) for x in example)
