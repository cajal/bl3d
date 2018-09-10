""" Importing structural data from our pipeline. """
import datajoint as dj
import numpy as np
from scipy import ndimage
from skimage import feature, morphology, measure, segmentation
stack = dj.create_virtual_module('stack', 'pipeline_stack') # only needed when populating data

from . import utils


dj.config['external-bl3d'] = {'protocol': 'file', 'location': '/mnt/scratch07/ecobost'}
dj.config['cache'] = '/tmp/dj-cache'
schema = dj.schema('ecobost_bl3d2', locals())


def get_stack(key, channel=1):
    """ Fetch a stack from the stack pipeline.

    Arguments:
        key: key used to constrain stack.CorrectedStack()
        channel: What channel to fetch. Starts at 1
    Returns:
        A 3-d np.array (depth x height x width). The stack.
    """
    slice_rel = (stack.CorrectedStack.Slice() & key & {'channel': channel})
    slices = slice_rel.fetch('slice', order_by='islice')
    return np.stack(slices)


@schema
class PreprocessingParams(dj.Lookup):
    definition = """ # manually tuned params to obtain instances from red channel

    -> stack.CorrectedStack
    ---
    bbox:               tinyblob # (pixels) bbox (min_z, max_z, min_y, max_y, min_x, max_x) to crop
    min_distance:       int      # minimum distance between peaks of local maxima in the volume
    threshold:          float    # threshold for final red label
    min_voxels:         int      # masks with less voxels than this are discarded
    max_voxels:         int      # masks with more voxels than this are discarded
    """
    contents = [
        [17026, 20, 1, 1, 1, (0, -1, 5, -9, 18, -5), 3, 0.07, 65, 2145],
        [17026, 20, 2, 1, 1, (0, -1, 4, -5, 15, -10), 3, 0.07, 65, 2145],
        [17206, 3, 1, 1, 1, (30, -1, 19, -10, 23, -14), 3, 0.07, 65, 2145],
        [17206, 3, 2, 1, 1, (0, -1, 19, -7, 26, -7), 3, 0.07, 65, 2145],
        [17206, 3, 7, 1, 1, (0, -1, 25, -13, 32, -10), 3, 0.07, 65, 2145],
        [17261, 2, 1, 1, 1, (0, -1, 5, -6, 10, -8), 3, 0.07, 65, 2145],
        [17261, 2, 2, 1, 1, (0, -1, 7, -8, 12, -8), 3, 0.07, 65, 2145],
        [17261, 2, 3, 1, 1, (0, -1, 9, -18, 17, -7), 3, 0.07, 65, 2145],
        [17259, 7, 2, 1, 1, (0, -1, 5, -5, 12, -8), 3, 0.07, 65, 2145],
        [17259, 7, 4, 1, 1, (0, -1, 7, -7, 13, -7), 3, 0.07, 65, 2145],
        [17259, 7, 6, 1, 1, (1, -6, 6, -5, 15, -7), 3, 0.07, 65, 2145],
        [17259, 7, 8, 1, 1, (0, -1, 6, -6, 17, -7), 3, 0.07, 65, 2145],
        [17259, 7, 10, 1, 1, (14, -15, 5, -5, 11, -5), 3, 0.07, 65, 2145],
    ]


@schema
class Stack(dj.Computed):
    definition = """ # a single stack with 3-d green and red structural recordings
    example_id:     mediumint
    ---
    -> PreprocessingParams
    """
    @property
    def key_source(self):
        return stack.CorrectedStack() & PreprocessingParams() # restrict to reso watermelon mice

    class Volume(dj.Part):
        definition = """ # GCaMP6 structural stack at 1 x 1 x 1 mm resolution
        -> master
        ---
        volume:         external-bl3d       # depth(z) x height (y) x width (x)
        """

    class Label(dj.Part):
        definition = """ # mask labels from the red channel (ids start at one, zero for background)
        -> master
        ---
        label:          external-bl3d        # depth(z) x height (y) x width (x)
        """

    class MaskProperties(dj.Part):
        definition = """ # some properties for each mask belonging to this label
        -> master
        mask_id:        int                 # mask id (same as in label)
        ---
        binary_mask:    blob                # binary 3-d mask (depth x height x width)
        volume:         int                 # number of voxels defined in this mask
        z_centroid:     float               # centroid of mask
        y_centroid:     float               # centroid of mask
        x_centroid:     float               # centroid of mask
        z_coord:        int                 # z-coordinate (first axis) of the corner closer to 0, 0, 0
        y_coord:        int                 # y-coordinate (second axis) of the corner closer to 0, 0, 0
        x_coord:        int                 # x-coordinate (third axis) of the corner closer to 0, 0, 0
        depth:          int                 # size in z
        height:         int                 # size in y
        width:          int                 # size in x
        mean_green:     float               # mean green channel intensity in the mask
        mean_red:       float               # mean red channel intensity in the mask
        """

    def make(self, key):
        # Get next example_id
        example_id = np.max(Stack.fetch('example_id')) + 1 if Stack() else 1
        self.insert1({'example_id': example_id, **key})

        print('Creating example', example_id, 'from', key)

        # Get resolution (microns per pixel)
        dims = (stack.CorrectedStack & key).fetch1()
        um_per_px = (dims['um_depth'] / dims['px_depth'], dims['um_height'] / dims['px_height'],
                     dims['um_width'] / dims['px_width'])

        # Get preprocessing params
        params = (PreprocessingParams & key).fetch1()
        bbox = (slice(params['bbox'][0], params['bbox'][1]),
                slice(params['bbox'][2], params['bbox'][3]),
                slice(params['bbox'][4], params['bbox'][5]))

        # Trim black edges and resize green channel to 1 x 1 x 1 mm^3 voxels
        ch1 = get_stack(key, channel=1)
        volume = ndimage.zoom(ch1[bbox], um_per_px, order=1, output=np.float32)
        self.Volume.insert1({'example_id': example_id, 'volume': volume})

        # Trim, resize, normalize contrast (locally) and smooth red channel
        ch2 = get_stack(key, channel=2)
        resized = ndimage.zoom(ch2[bbox], um_per_px, order=1)
        norm = utils.lcn(resized, (1.3, 13, 13))

        # Find blobs at different scales
        log1 = -ndimage.gaussian_laplace(norm, (3.5, 2.8, 2.8))
        log3 = -ndimage.gaussian_laplace(norm, (4.75, 3.8, 3.8))
        log2 = -ndimage.gaussian_laplace(norm, (6, 4.8, 4.8))
        log4 = -ndimage.gaussian_laplace(norm, (7.25, 5.8, 5.8))
        blobs = np.maximum(np.maximum(np.maximum(log1, log2), log3), log4)

        # Create masks
        peaks = feature.peak_local_max(blobs, exclude_border=False, indices=False,
                                       footprint=morphology.ball(params['min_distance']))
        markers = morphology.label(peaks)
        masks = morphology.watershed(-blobs, markers, mask=(blobs > params['threshold']),
                                     connectivity=3)

        # Remove masks that are too small or too big (usually bad detections)
        mask_sizes = np.bincount(masks.ravel())
        to_keep = np.logical_and(mask_sizes >= params['min_voxels'],
                                 mask_sizes <= params['max_voxels'])
        masks[~to_keep[masks]] = 0 # set to background
        label, _, _ = segmentation.relabel_sequential(masks)

        # Insert
        self.Label.insert1({'example_id': example_id, 'label': label.astype(np.uint16)})

        # Save some mask properties
        properties = measure.regionprops(label, intensity_image=resized, cache=False)
        for mask in properties:
            centroid = mask.centroid
            bbox = mask.bbox
            green_box = volume[bbox[0]: bbox[3], bbox[1]: bbox[4], bbox[2]: bbox[5]]
            self.MaskProperties.insert1({'example_id': example_id, 'mask_id': mask.label,
                'binary_mask': mask.image, 'volume': mask.area, 'z_centroid': centroid[0],
                'y_centroid': centroid[1], 'x_centroid': centroid[2], 'z_coord': bbox[0],
                'y_coord': bbox[1], 'x_coord': bbox[2], 'depth': bbox[3] - bbox[0],
                'height': bbox[4] - bbox[1], 'width': bbox[5] - bbox[2],
                'mean_red': mask.mean_intensity, 'mean_green': green_box[mask.image].mean()
            })