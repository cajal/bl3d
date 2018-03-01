""" Some simple 3-D transforms. """
import numpy as np

class RandomCrop:
    """ Randomly crop a 3-d patch from a volume.

    Arguments:
            size: int or triple. Size of the crop in depth, height, width.
    """
    def __init__(self, size=256):
        self.size = size if isinstance(size, tuple) else (size,) * 3

    def __call__(self, example):
        """ Apply transform.

        Arguments:
        example: (volume, label) tuple. A 4-d and 3-d numpy array: channels x depth x
            height x width and depth x height x width. Does NOT work in tensors.
        """
        max_indices = [max(0, dim - siz) for dim, siz in zip(example[1].shape, self.size)]
        start = [np.random.randint(max_idx + 1) for max_idx in max_indices]
        slices = [slice(st, st + siz) for st, siz in zip(start, self.size)]

        return tuple(x[..., slices[0], slices[1], slices[2]] for x in example)


class RandomRotate:
    """ Rotates an example (volume, label) by 0, 90, 180 or 270 degrees in the z axis."""
    def __call__(self, example):
        """ Apply transform.

        Arguments:
        example: (volume, label) tuple. A 4-d and 3-d numpy array: channels x depth x
            height x width and depth x height x width. Does NOT work in tensors.
        """
        k = np.random.choice([0, 1, 2, 3]) # number of 90 degree rotations
        return tuple(np.rot90(x, k, axes=(-2, -1)) for x in example)


class RandomHorizontalFlip:
    """ Mirrors the example (volume, label) in the horizontal axis with p=0.5.

    Note: Because we also rotate at 0, 90, 180 and 270, vertical flipping is redundant.
        For instance, rot0 + flip_v = rot180 + flip_h
    """
    def __call__(self, example):
        """ Apply transform.

        Arguments:
        example: (volume, label) tuple. A 4-d and 3-d numpy array: channels x depth x
            height x width and depth x height x width. Does NOT work in tensors.
        """
        flip = np.random.random() > 0.5 # flip with equal probability
        return tuple(np.flip(x, axis=-1) for x in example) if flip else example


class ContrastNorm:
    """ Normalizes the volume per channel to zero mean and stddev=1. Label is untouched."""
    def __call__(self, example):
        """ Apply transform.

        Arguments:
        example: (volume, label) tuple. A 4-d and 3-d numpy array: channels x depth x
            height x width and depth x height x width. Does NOT work in tensors.
        """
        volume = ((example[0] - example[0].mean(axis=(-1, -2, -3))) /
                  example[0].std(axis=(-1, -2, -3)))
        return (volume, example[1])

class Copy:
    """ Creates a copy of the example.

    See: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    """
    def __call__(self, example):
        """ Apply transform.

        Arguments:
        example: (volume, label) tuple. A 4-d and 3-d numpy array: channels x depth x
            height x width and depth x height x width. Does NOT work in tensors.
        """
        return tuple(x.copy() for x in example)