""" Some utility functions. """
import numpy as np


def create_video(stack, interval=100, repeat_delay=1000):
    """ Create an animation of the stack. Fly-through depth axis.

    Arguments:
        stack (np.array): A depth x height x width stack.
        interval (int): Number of milliseconds between frames.
        repeat_delay (int): Number of milliseconds to wait at end of presentation before
            starting a new presentation.

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
    return fig, video  # if video is garbage collected, the animation stops


def colorize_label(label, num_colors=100):
    """ Transform single int labels into RGB with random colors.

    Arguments:
        label (np.array): Labelled voxels with zero for background and positive integers
            for detected instances.
        num_colors (int): Number of random colors to use.

    Returns:
        Array (*label_shape x 3). RGB volume with a different random color for each
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
            pass  # other non-tensor objects


def lcn(image, sigmas=(12, 12)):
    """ Local contrast normalization. Normalize each pixel using mean and stddev computed
    on a local neighborhood.

    We use gaussian filters rather than uniform filters to compute the local mean and std
    to soften the effect of edges; essentially we use a fuzzy local neighborhood.

    Arguments:
        image (np.array): Raw two-photon stack.
        sigmas (tuple): Sigmas to use for the gaussian filter (one per axis). Smaller
            values result in more local neighborhoods. 15-30 microns should work fine
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
        image (np.array): Raw two-photon stack.
        laplace_sigma (float): Sigma of the gaussian used in the laplace filter.
        low_percentile (float): Lower percentile used to clip intensities.
        high_percentile (float): Higher percentile used to clip intensities.

    Returns:
        Array of same shape as input. Sharpened image.
    """
    from scipy import ndimage

    sharpened = image - ndimage.gaussian_laplace(image, laplace_sigma)
    clipped = np.clip(sharpened, *np.percentile(sharpened, [low_percentile,
                                                            high_percentile]))
    norm = (clipped - clipped.mean()) / (clipped.max() - clipped.min() + 1e-7)

    return norm


def prob2labels(detection, segmentation_, seg_threshold=0.5, min_voxels=65,
                max_voxels=4589):
    """ Create instance segmentations using centroid predictions and cell segmentations.

    Arguments:
        detection (np.array): 3-d probability heatmap.
        segmentation (np.array): 3-d probability heatmap.
        threshold (np.array): Threshold for the segmentation heatmap.
        min_voxels (int): Minimum number of voxels a final mask would have.
        max_voxels (int): Maximum number of voxels a final mask would have

    Returns:
        label: Array with same shape as segmentation with zero for background and positive
            integer ids for each predicted instance.
    """
    from skimage import feature, morphology, measure, segmentation

    # Create binary segmentation
    binary_masks = morphology.remove_small_holes(segmentation_ > seg_threshold)

    # Watershed segmentation using centroids from detection
    peaks = feature.peak_local_max(detection, footprint=morphology.ball(4), indices=False,
                                   labels=binary_masks, exclude_border=False)
    markers = morphology.label(peaks)
    masks = morphology.watershed(-segmentation_, markers, mask=binary_masks,
                                 connectivity=3)
    print(masks.max(), 'initial cells')

    # Remove masks that are too small or too big (usually bad detections)
    mask_sizes = np.bincount(masks.ravel())
    to_keep = np.logical_and(mask_sizes >= min_voxels, mask_sizes <= max_voxels)
    masks[~to_keep[masks]] = 0  # set to background
    label, _, _ = segmentation.relabel_sequential(masks)
    print(label.max(), 'final cells')

    return label