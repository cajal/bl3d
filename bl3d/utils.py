""" Some utility functions. """
import numpy as np


def create_video(stack, interval=100, repeat_delay=1000):
    """ Create an animation of the stack. Fly-through depth axis.

    Arguments:
        stack: 3-d array (depth x height x width)
        interval: Number of milliseconds between frames
        repeat_delay: Number of milliseconds to wait at end of presentation before
            starting new one.

    Returns:
        Figure and video handle.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig = plt.figure()
    num_slices = stack.shape[0]
    im = fig.gca().imshow(stack[int(num_slices / 2)])
    def update_img(i):
        im.set_data(stack[i])
    video = animation.FuncAnimation(fig, update_img, num_slices, interval=interval,
                                    repeat_delay=repeat_delay)
    # video.save('my_video.mp4', dpi=250)
    return fig, video # if video is garbage collected, the animation stops


def colorize_label(label, num_colors=100):
    """ Transform single int labels into RGB with random colors.

    Arguments:
        label: Array with zero for background and positive integers for each detected
            instance.
        num_colors: Number of random colors to use.

    Returns:
        Array (*label_shape x 3). RGB image/volume with a different random color for each
        instance in label.
    """
    from skimage import color

    colors = np.random.randint(0, 256, size=(num_colors, 3))
    rgb = color.label2rgb(label, colors=colors, bg_label=0)
    rgb = rgb.astype(np.uint8)

    return rgb


def memusage():
    """ Print a report  of all Tensor objects in memory and their size. """
    # Source: https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/3
    import torch, gc

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                print(obj.type(), obj.shape)
        except:
            pass # other non-tensor objects


def lcn(image, sigmas=(12, 12)):
    """ Local contrast normalization.

    Normalize each pixel using mean and stddev computed on a local neighborhood.

    We use gaussian filters rather than uniform filters to compute the local mean and std
    to soften the effect of edges. Essentially we are using a fuzzy local neighborhood.
    Using a hard definition of neighborhood, the equivalent will be:
        local_mean = ndimage.uniform_filter(image, size=(32, 32))

    :param np.array image: Array with raw two-photon images.
    :param tuple sigmas: List with sigmas per axes to use for the gaussian filter.
        Smaller values result in more local neighborhoods. 15-30 microns should work fine
    """
    from scipy import ndimage

    local_mean = ndimage.gaussian_filter(image, sigmas)
    local_std = np.sqrt(ndimage.gaussian_filter(image ** 2, sigmas) -
                        ndimage.gaussian_filter(image, sigmas) ** 2)
    norm = (image - local_mean) / (local_std + 1e-7)

    return norm


def sharpen_2pimage(image, laplace_sigma=0.7, low_percentile=3, high_percentile=99.9):
    """ Apply a laplacian filter, clip pixel range and normalize.

    Arguments:
        image: Array with raw two-photon images.
        laplace_sigma: Sigma of the gaussian used in the laplace filter.
        low_percentile, high_percentile: Percentiles at which to clip.

    Returns:
        Array of same shape as input. Sharpened image.
    """
    from scipy import ndimage

    sharpened = image - ndimage.gaussian_laplace(image, laplace_sigma)
    clipped = np.clip(sharpened, *np.percentile(sharpened, [low_percentile, high_percentile]))
    norm = (clipped - clipped.mean()) / (clipped.max() - clipped.min() + 1e-7)

    return norm


def combine_masks(shape, masks, bboxes, use_ids=True):
    """ Combines masks into a single 3-d volume using bbox coordinates.

    If two masks are defined in the same voxel, we select the one that appears first in
    the list.

    Arguments:
        shape: A tuple. Shape of the full volume.
        masks: A list of arrays. The masks to combine.
        bboxes: NMASKS x 2*DIM array. The bbox coordinates of each mask (z, y, x, ..., d, h, w, ...).
        use_ids: Boolean. Set voxels where mask is positive to the index of the mask.

    Returns:
        An array with all masks in their respective position.
    """
    # Compute masks indices (where to paste them in the output volume)
    dim = bboxes.shape[1] // 2
    low_indices = np.round(bboxes[:, :dim] - bboxes[:, dim:] / 2).astype(int) # N x dim
    high_indices = np.round(bboxes[:, :dim] + bboxes[:, dim:] / 2).astype(int) # N x dim
    mask_slices = [tuple(slice(np.clip(-l, 0, h - l), h - l - np.clip(h - s, 0, h - l))
                         for l, h, s in zip(low, high, shape)) for low, high in
                   zip(low_indices, high_indices)]
    full_slices = [tuple(slice(max(l, 0), max(h, 0)) for l, h in zip(low, high)) for
                   low, high in zip(low_indices, high_indices)]

    # Create combined volume
    volume = np.zeros(shape, dtype=(int if use_ids else float))
    for i, mask, sl, mask_sl in zip(range(len(masks), -1, -1), masks[::-1],
                                    full_slices[::-1], mask_slices[::-1]):
        if use_ids:
            volume[sl][mask[mask_sl]] = i + 1 # indices start at 1
        else:
            volume[sl] = mask[mask_sl]

    return volume