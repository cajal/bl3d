""" Importing structural data from our pipeline. """
import datajoint as dj
import numpy as np
from scipy import ndimage
from skimage import feature, morphology, measure, segmentation
stack = dj.create_virtual_module('stack', 'pipeline_stack') # only needed when populating data

from . import utils


dj.config['external-bl3d'] = {'protocol': 'file', 'location': '/mnt/lab/users/ecobost'}
dj.config['cache'] = '/tmp/dj-cache'
schema = dj.schema('ecobost_bl3d', locals())



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
    definition = """ # some manually tuned params for processing the red label

    -> stack.CorrectedStack
    ---
    bbox:               tinyblob # bbox (min_z, max_z, min_y, max_y, min_x, max_x) to crop
    gaussian_std:       float    # stddev of the smoothing filter
    threshold:          float    # threshold for the red label
    min_distance:       int      # minimum distance between local maxima in the volume
    min_voxels:         int      # masks with less voxels than this are discarded
    max_voxels:         int      # masks with more voxels than this are discarded
    """
    contents = [ # animal_id, session, stack_idx, pipe_version, volume_id, bbox, gaussian_std, ...
        [17026, 20, 1, 1, 1, (0, -1, 2, -5, 5, -2), 0.6, 1.4, 5, 113, 4189],
        [17026, 20, 2, 1, 1, (0, -1, 1, -3, 5, -2), 0.6, 1.4, 5, 113, 4189],
        [17206, 3, 1, 1, 1, (15, -1, 15, -7, 15, -8), 0.6, 1.3, 5, 113, 4189],
        [17206, 3, 2, 1, 1, (0, -1, 15, -5, 15, -2), 0.6, 1.3, 5, 113, 4189],
        [17206, 3, 7, 1, 1, (0, -1, 21, -5, 11, -4), 0.6, 1.3, 5, 113, 4189],
        [17261, 2, 1, 1, 1, (0, -1, 5, -4, 9, -5), 0.6, 1.3, 5, 113, 4189],
        [17261, 2, 2, 1, 1, (1, -1, 5, -5, 10, -5), 0.6, 1.3, 5, 113, 4189],
        [17261, 2, 3, 1, 1, (1, -1, 5, -6, 10, -5), 0.6, 1.3, 5, 113, 4189],
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

    class EnhancedVolume(dj.Part):
        definition = """ # volume after local contrast normalization and laplacian sharpening
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

    def make(self, key):
        # Get next example_id
        example_id = np.max(Stack().fetch('example_id')) + 1 if Stack() else 1
        self.insert1({'example_id': example_id, **key})

        print('Creating example', example_id, 'from', key)

        # Get resolution (microns per pixel)
        dims = (stack.CorrectedStack() & key).fetch1()
        um_per_px = (dims['um_depth'] / dims['px_depth'], dims['um_height'] / dims['px_height'],
                     dims['um_width'] / dims['px_width'])

        # Get preprocessing params
        params = (PreprocessingParams() & key).fetch1()
        bbox = (slice(params['bbox'][0], params['bbox'][1]),
                slice(params['bbox'][2], params['bbox'][3]),
                slice(params['bbox'][4], params['bbox'][5]))

        # Resize green channel to 1 x 1 x 1 mm^3 voxels and trim black edges
        green = get_stack(key, channel=1)
        volume = ndimage.zoom(green, um_per_px, order=1, output=np.float32)[bbox]
        self.Volume().insert1({'example_id': example_id, 'volume': volume})

        # Save enhanced green channel (local contrast normalization -> sharpening)
        enhanced = utils.sharpen_2pimage(utils.lcn(volume, (3, 30, 30)))
        self.EnhancedVolume().insert1({'example_id': example_id, 'volume': enhanced})

        # Resize, crop, normalize contrast (locally) and smooth red channel
        red = get_stack(key, channel=2)
        resized = ndimage.zoom(red, um_per_px, order=1)[bbox]
        mean = ndimage.uniform_filter(resized, (3, 30, 30))
        stddev = np.sqrt(ndimage.uniform_filter(resized ** 2, (3, 30, 30)) -
                         ndimage.uniform_filter(resized, (3, 30, 30)) ** 2)
        enhanced = ndimage.gaussian_filter((resized - mean) / stddev, params['gaussian_std'])

        # Get masks (watershed segmentation from local maxima)
        np.random.seed(123)
        thresholded = enhanced > params['threshold']
        filled = morphology.remove_small_objects(morphology.remove_small_holes(thresholded),
                                                 params['min_voxels'])
        distance = ndimage.distance_transform_edt(filled)
        distance += 1e-7 * np.random.random(distance.shape) # to break ties
        peaks = feature.peak_local_max(distance, min_distance=params['min_distance'],
                                       labels=filled, indices=False)
        markers = morphology.label(peaks)
        label = morphology.watershed(-distance, markers, mask=filled)

        # Remove masks that are too small or too large
        label = morphology.remove_small_objects(label, params['min_voxels'])
        too_large = [p.label for p in measure.regionprops(label) if p.area > params['max_voxels']]
        for label_id in too_large:
            label[label == label_id] = 0 # set to background
        label, _, _ = segmentation.relabel_sequential(label)
        self.Label().insert1({'example_id': example_id, 'label': label.astype(np.int32)})


@schema
class AverageCell(dj.Computed):
    definition = """ # mean image of the cells
    -> Stack
    ---
    volume:             longblob            # 31 x 31 x 31 mean image in the green channel
    label:              longblob            # 31 x 31 x 31 mean image in the red channel
    """
    def make(self, key):
        print('Populating key', key)

        # Get data
        volume = (Stack.Volume() & key).fetch1('volume')
        label = (Stack.Label() & key).fetch1('label')
        zs, ys, xs = (Stack.MaskProperties() & key).fetch('z_centroid', 'y_centroid', 'x_centroid')
        zs, ys, xs = zs.round().astype(int), ys.round().astype(int), xs.round().astype(int)

        # Create mean images
        green = np.zeros((31, 31, 31))
        red = np.zeros((31, 31, 31))
        num_cells_in_range = 0
        for z, y, x in zip(zs, ys, xs):
            if (z - 15 >= 0 and z + 15 < volume.shape[0] and y - 15 >= 0 and
                y + 15 < volume.shape[1] and x - 15 >= 0 and x + 15 < volume.shape[2]):
                green += volume[z - 15:z + 15 + 1, y - 15:y + 15 + 1, x - 15:x + 15 + 1]
                red += label[z - 15:z + 15 + 1, y - 15:y + 15 + 1, x - 15:x + 15 + 1]
                num_cells_in_range += 1
        green /= num_cells_in_range
        red /= num_cells_in_range

        # Insert
        self.insert1({**key, 'volume': green, 'label': red})