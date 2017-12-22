""" Hyperparameter configurations. """
import datajoint as dj
import itertools

from bl3d import utils


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
    """
    items = itertools.product(
        [1e-4, 1e-3, 1e-2, 1e-1],           # learning_rate
        [0, 1e-5, 1e-3, 1e-1, 1e1],         # weight decay
        [1234],                             # seed
        [200],                              # num_epochs
        [0.9],                              # momentum
        [0.95],                             # lr_decay
        ['val_loss']                        # lr_schedule: could be 'none', every 'epoch' or epochs when 'val_loss' does not decrease
    )
    contents = [[utils.list_hash(item), *item] for item in items]


@schema
class ModelParams(dj.Lookup):
    definition = """ # different models to train
    model_hash:             varchar(16)     # unique id for network configurations
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
        num_features:       int             # number of feature maps per layer
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
        num_conv_layers:    tinyint         # number of convolutional networks
        num_fc_layers:      tinyint         # number of fully connected networks (after conv)
        num_features:       tinyblob        # number of feature maps per layer
        kernel_sizes:       tinyblob        # list with kernel sizes (one per conv layer)
        dilation:           tinyblob        # list with dilation (one per conv layer)
        padding:            tinyblob        # list with padding amounts (one per conv layer)
        use_batchnorm:      boolean         # whether to use batch_normalization
        """
        hash_prefix = 'fcn'
        items = itertools.product(
            [5],                            # num_conv_layers
            [2],                            # num_fc_layers
            [(1, 8, 8, 16, 16, 32, 32, 2)], # num_features
            [(3, 3, 3, 3, 3)],              # kernel_sizes
            [(1, 1, 2, 2, 3)],              # dilation
            [(1, 1, 2, 2, 3)],              # padding
            [False]                         # use_batchnorm
        )

    def fill():
        for model in [Linear, Dictionary, FCN]:
            for item in model.items:
                model_hash = model.hash_prefix + '_' + utils.list_hash(item)
                ModelParams.insert1({'model_hash': model_hash}, skip_duplicates=True)
                model.insert1([model_hash, *item], skip_duplicates=True)

    def build_model(key):
        """ Construct model with the required configuration. """
        raise NotImplementedError


class EvalParams(dj.Lookup):
    definition=""" # different parameters used during evaluation
    eval_hash:              varchar(64)     # unique id for evaluation parameters
    ---
    gaussian_std:           float           # standard deviation of gaussian window used for smoothing
    use_gradimage:          boolean         # whether to use the gradient of the predictions
    integrate_probs:        boolean         # whether to
    num_thresholds:         tinyint         # number of (linspaced) thresholds to try
    min_voxels:             int             # minimum number of voxels for a mask to be valid
    max_voxels:             int             # maximum number of voxels for a mask to be valid
    """

    items = itertools.product(
        [0, 0.7],                           # gaussian_std
        [False, True],                      # use_grad_image
        [26],                               # num_thresholds
        [34, 113, 268],                     # min_voxels: sphere volume for d in [4, 6, 8]
        [1437, 2144, 3054, 1e9],            # max_voxels: sphere volume fo r d in [14, 16, 18, infty]
    )
    contents = [[utils.list_hash(item), *item] for item in items]