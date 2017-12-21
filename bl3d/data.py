""" Importing structural data from our pipeline. """
import datajoint as dj
import numpy as np
from scipy import ndimage
from skimage import feature, morphology, measure, segmentation
stack = dj.create_virtual_module('stack', 'pipeline_stack') # only needed when populating data



dj.config['external-bl3d'] = {'protocol': 'file', 'location': '/mnt/lab/ecobost'}
dj.config['cache'] = '/home/ecobost/dj-cache'
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
class WatermelonMice(dj.Lookup):
    definition=""" # animals with red nuclear labels
    animal_id:          int
    """
    contents = [[17206], [17026]]
    #contents = [[16271], [16278], [16612], [17026], [17206], [17264]] # meso scans


@schema
class PreprocessingParams(dj.Lookup):
    definition = """ # some manually tuned params for processing the red label

    -> stack.CorrectedStack
    ---
    gaussian_std:       float    # stddev of the smoothing filter
    threshold:          float    # threshold for the red label
    min_distance:       int      # minimum distance between local maxima in the volume
    min_voxels:         int      # masks with less voxels than this are discarded
    max_voxels:         int      # masks with more voxels than this are discarded
    """
    contents = [ # animal_id, session, stack_idx, pipe_version, volume_id, gaussian_std, ...
        [17206, 3, 1, 1, 0.7, 1.45, 5, 113, 4189],
        [17206, 3, 2, 1, 0.7, 1.45, 5, 113, 4189],
        [17206, 3, 7, 1, 0.7, 1.45, 5, 113, 4189],
        [17026, 20, 1, 1, 0.7, 1.45, 5, 113, 4189],
        [17026, 20, 2, 1, 0.7, 1.45, 5, 113, 4189],
    ]


@schema
class Stack(dj.Computed):
    definition = """ # a single stack with 3-d green and red structural recordings
    example_id:     mediumint
    ---
    -> stack.CorrectedStack
    """
    @property
    def key_source(self):
        return stack.CorrectedStack() & PreprocessingParams()

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
        mean_red:       float               # mean red channel intensity in the mask
        mean_green:     float               # mean green channel intensity in the mask
        """

    def make(self, key):
        # Get next example_id
        example_id = np.max(Stack().fetch('example_id')) + 1 if Stack() else 1
        key['example_id'] = example_id

        print('Creating example', key)

        # Get microns per pixel resolution
        dims = (stack.CorrectedStack() & key).fetch1()
        um_per_px = (dims['um_depth'] / dims['px_depth'], dims['px_height'] / dims['um_height'],
                     dims['um_width'] / dims['px_width'])

        # Process green channel: resize to 1 x 1 x 1 mm^3 voxels
        green = get_stack(key, channel=1)
        volume = ndimage.zoom(green, um_per_px, order=1)
        self.Volume().insert1({'key': key, 'volume': volume})

        # Process red channel: resize, balance brightness across FOV and smooth a little to denoise
        params = (PreprocessingParams() & key).fetch1()
        red = get_stack(key, channel=2)
        resized = ndimage.zoom(red, um_per_px, order=1)
        background = ndimage.gaussian_filter(resized, (3, 20, 20))
        enhanced = ndimage.gaussian_filter(resized / background, params['gaussian_std'])

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
        self.Label().insert1({**key, 'label': label})

        # Save some mask properties
        properties = measure.regionprops(label, intensity_image=red, cache=False)
        for mask in properties:
            centroid = mask.centroid
            bbox = mask.bbox
            green_box = green[bbox[0]: bbox[3]-bbox[0], bbox[1]: bbox[4]-bbox[1],
                              bbox[2]: bbox[5]-bbox[2]]
            self.MaskProperties().insert1({**key, 'mask_id': mask.label,
                'binary_mask': mask.image, 'volume': mask.area, 'z_centroid': centroid[0],
                'y_centroid': centroid[1], 'x_centroid': centroid[2], 'z_coord': bbox[0],
                'y_coord': bbox[1], 'x_coord': bbox[2], 'depth': bbox[3] - bbox[0],
                'height': bbox[4] - bbox[1], 'width': bbox[5] - bbox[2],
                'mean_red': mask.mean_intensity,'mean_green': green_box[mask.image].mean()
                })