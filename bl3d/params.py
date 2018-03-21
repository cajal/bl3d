""" Hyperparameter configurations. """
import datajoint as dj
import itertools

from bl3d import utils
from bl3d import models


schema = dj.schema('ecobost_bl3d', locals())


@schema
class TrainingParams(dj.Lookup):
    definition = """ # different hyperparameters to search over
    training_hash:          varchar(64)     # unique id for hyperparameter combination
    ---
    learning_rate:          float           # initial learning rate for SGD
    weight_decay:           float           # lambda for l2-norm weight regularization
    seed:                   int             # random seed for torch.manual_seed()
    num_epochs:             int             # number of training epochs
    momentum:               float           # momentum factor for SGD updates
    lr_decay:               float           # factor to multiply learning rate every epoch
    lr_schedule:            varchar(8)      # type of learning rate decay to use
    positive_weight:        float           # relative weight for positive class examples (negative class weight is 1)
    """
    items = itertools.product(
        [1e-4, 1e-3, 1e-2, 1e-1],           # learning_rate
        [0, 1e-5, 1e-3, 1e-1, 1e1],         # weight decay
        [1234],                             # seed
        [100],                              # num_epochs
        [0.9],                              # momentum
        [0.95],                             # lr_decay
        ['val_loss', 'none'],               # lr_schedule: could be 'none', every 'epoch' or epochs when 'val_loss' does not decrease
        [1, 4]                              # positive_weight
    )
    contents = []
    for item in items:
        contents.append([utils.list_hash(item[:-1]), *item] if item[-1] == 1 else [utils.list_hash(item), *item])


@schema
class ModelParams(dj.Lookup):
    definition = """ # different models to train
    model_hash:             varchar(64)     # unique id for network configurations
    """
    class Linear(dj.Part):
        definition = """ # single 3-d linear filter (plus softmax)
        -> master
        ---
        filter_size:        tinyblob        # size of the filter
        num_features:       tinyblob        # number of feature maps per layer
        """
        hash_prefix = 'linear'
        items = itertools.product(
            [(25, 19, 19)],                 # filter_size: (depth, height, width)
            [(1, 2)],                       # num_features: (in_channels, out_channels)
        )

    class Dictionary(dj.Part):
        definition = """ # different filters combined to produce a prediction
        -> master
        ---
        filter_size:        tinyblob        # size of the filters
        num_features:       tinyblob        # number of feature maps per layer
        use_batchnorm:      boolean         # whether to use batch normalization
        """
        hash_prefix = 'dict'
        items = itertools.product(
            [(25, 19, 19)],                 # filter_size: (depth, height, width)
            [(1, 16, 2)],                   # num_features: (in_channels, num_filters, out_channels)
            [False]                         # use_batchnorm
        )

    class FCN(dj.Part):
        definition = """ # a fully convolutional network for segmentation
        -> master
        ---
        num_features:       tinyblob        # number of feature maps per layer
        kernel_sizes:       tinyblob        # list with kernel sizes (one per conv layer)
        dilation:           tinyblob        # list with dilation (one per conv layer)
        padding:            tinyblob        # list with padding amounts (one per conv layer)
        use_batchnorm:      boolean         # whether to use batch normalization
        """
        hash_prefix = 'fcn'
        items = itertools.product(
            [(1, 8, 8, 16, 16, 32, 32, 2)], # num_features
            [(3, 3, 3, 3, 3, 1, 1)],        # kernel_sizes
            [(1, 1, 2, 2, 3, 1, 1)],        # dilation
            [(1, 1, 2, 2, 3, 0, 0)],        # padding
            [False, True]                   # use_batchnorm
        )

    def fill():
        for model in [ModelParams.Linear, ModelParams.Dictionary, ModelParams.FCN]:
            for item in model.items:
                model_hash = model.hash_prefix + '_' + utils.list_hash(item)
                ModelParams.insert1({'model_hash': model_hash}, skip_duplicates=True)
                model.insert1([model_hash, *item], skip_duplicates=True)

    def build_model(key):
        """ Construct model with the required configuration. """
        if ModelParams.Linear() & key:
            params = (ModelParams.Linear() & key).fetch1('num_features', 'filter_size')
            return models.LinearFilter(*params)
        elif ModelParams.Dictionary() & key:
            params = (ModelParams.Dictionary() & key).fetch1('num_features', 'filter_size',
                                                             'use_batchnorm')
            return models.Dictionary(*params)
        elif ModelParams.FCN() & key:
            params = (ModelParams.FCN() & key).fetch1('num_features', 'kernel_sizes',
                                                      'dilation', 'padding', 'use_batchnorm')
            return models.FullyConvNet(*params)
        else:
            raise ValueError('Model key {} not found.'.format(key))


@schema
class CrossValParams(dj.Lookup):
    definition=""" # different parameters used during evaluation
    xval_hash:              varchar(64)     # unique id for evaluation parameters
    ---
    num_thresholds:         tinyint         # number of (linspaced) thresholds to try
    gaussian_std:           float           # standard deviation of gaussian window used for smoothing
    min_distance:           int             # minimum distance between local maxima in the volume
    min_voxels:             int             # minimum number of voxels for a mask to be valid
    max_voxels:             int             # maximum number of voxels for a mask to be valid
    """

    items = itertools.product(
        [26],                               # num_thresholds
        [0, 0.6],                           # gaussian_std
        [4, 5, 6],                          # min_distance
        [34, 113, 268],                     # min_voxels: sphere volume for d in [4, 6, 8]
        [1437, 2144, 3054, 1e9],            # max_voxels: sphere volume fo r d in [14, 16, 18, infty]
    )
    contents = [[utils.list_hash(item), *item] for item in items]