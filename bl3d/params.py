""" Hyperparameter configurations. """
import datajoint as dj


schema = dj.schema('ecobost_bl3d3', locals())


@schema
class TrainingParams(dj.Lookup):
    definition = """ # different hyperparameters to search over

    training_id:        int             # unique id for hyperparameter combination
    ---
    seed:               int             # random seed for torch/np      
    normalize_volume:   boolean         # whether to local contrast normalize the stacks
    centroid_radius:    int             # radius of the centroids to predict 
    train_crop_size:    int             # size of the crops used during training
    val_crop_size:      int             # size of the random crops used for validation
    learning_rate:      float           # initial learning rate for SGD
    momentum:           float           # momentum factor for SGD updates
    weight_decay:       float           # lambda for l2-norm weight regularization
    lr_decay:           float           # factor multiplying learning rate when decreasing
    num_epochs:         int             # number of training epochs
    val_epochs:         int             # run validation every this number of epochs
    decay_epochs:       int             # number of epochs to wait before decreasing learning rate if val loss has not improved 
    nsn_loss_weight:    float           # weight for the nsn loss (ndn loss has weight 1)
    ndn_pos_weight:     float           # relative weight for positive class examples in the NDN (negative class weight is 1)
    nsn_pos_weight:     float           # relative weight for positive class examples in the NSN (negative class weight is 1)
    ndn_threshold:      float           # percentile used as threshold when computing IOU
    nsn_threshold:      float           # percentile used as threshold when computing IOU 
    """
    @property
    def contents(self):
        import itertools
        id_ = 1

        for learning_rate, weight_decay, nsn_loss_weight, ndn_pos_weight, nsn_pos_weight \
            in itertools.product([0.01, 0.1], [0, 0.0001, 0.01], [0.1, 1, 10], [300],
                                 [10]):
            yield {'training_id': id_, 'seed': 1234, 'normalize_volume': True,
                   'centroid_radius': 2, 'train_crop_size': 128, 'val_crop_size': 192,
                   'learning_rate': learning_rate, 'momentum': 0.9,
                   'weight_decay': weight_decay, 'lr_decay': 0.1, 'num_epochs': 150,
                   'val_epochs': 1, 'decay_epochs': 10, 'ndn_pos_weight': ndn_pos_weight,
                   'nsn_pos_weight': nsn_pos_weight, 'nsn_loss_weight': nsn_loss_weight,
                   'ndn_threshold': 99.7, 'nsn_threshold': 91}
            id_ += 1


@schema
class DenseNet(dj.Lookup):
    definition = """ # core dense net

    core_version:       tinyint
    ---
    initial_maps:       tinyint     # number of feature maps in the initial layer
    layers_per_block:   blob        # number of layers in each dense block 
    growth_rate:        tinyint     # how many feature maps to add each layer
    compression_factor: float       # how to increase/decrease feature maps in transition layers
      
    """
    contents = [
        [1, 16, (6, 6), 8, 0.5]
    ]


@schema
class NDN(dj.Lookup):
    definition = """ # nuclear detection network

    ndn_version:        tinyint
    ---
    num_features:       tinyblob        # number of feature maps per layer
    kernel_sizes:       tinyblob        # list with kernel sizes (one per layer)
    dilation:           tinyblob        # list with dilation (one per layer)
    padding:            tinyblob        # list with padding (one per layer)
    """
    contents = [
        [1, (96, 1), (3, 1), (1, 1), (1, 0)]
    ]


@schema
class NSN(dj.Lookup):
    definition = """ # nuclear segmentation network

    nsn_version:       tinyint
    ---
        ---
    num_features:       tinyblob        # number of feature maps per layer
    kernel_sizes:       tinyblob        # list with kernel sizes (one per layer)
    dilation:           tinyblob        # list with dilation (one per layer)
    padding:            tinyblob        # list with padding (one per layer)
    """
    contents = [
        [1, (96, 1), (3, 1), (1, 1), (1, 0)]
    ]


@schema
class ModelParams(dj.Lookup):
    definition = """ # architectural details of our different models

    model_version:      tinyint         # unique id for this network
    ---
    -> DenseNet
    -> NDN
    -> NSN
    """
    contents = [
        [1, 1, 1, 1]
    ]


@schema
class TrainingSplit(dj.Lookup):
    definition = """ # examples used for training and validation (no test set)

    val_animal:         int         # animal_id used for validation
    ---
    num_examples:       int         # number of examples in this set
    train_examples:     longblob    # list of examples used for training
    val_examples:       longblob    # list of examples used in validation
    """
    @property
    def contents(self):
        from . import data

        example_ids = set(data.Stack.fetch('example_id'))
        num_examples = len(example_ids)
        for animal_id in set(data.Stack.fetch('animal_id')): # each animal
            val_examples = (data.Stack & {'animal_id': animal_id}).fetch('example_id')
            yield [animal_id, num_examples, example_ids - set(val_examples), val_examples]


# @schema
# class EvalParams(dj.Lookup):
#     definition=""" # parameters used during evaluation
#
#     eval_id:            int         # id for the evaluation parameters
#     ---
#     nms_iou:            float       # IOU used as non-maximum suppression threshold
#     cells_per_um :      float       # estimated number of cells per 1 um cube
#     proposal_factor:    float       # estimated cells * proposal_factor = num_proposals
#     mask_factor:        float       # estimated cells * masks_factor = num_masks
#     mask_threshold:     float       # probability threshold for mask binarization
#     """
#     contents = [
#         [1, 0.5, 0.0001, 3, 1.5, 0.5]
#     ]


@schema
class EvalSet(dj.Lookup):
    definition = """ # set where metrics are computed
    eval_set:    varchar(8)
    """
    contents = [
        ['train'],
        ['val']
    ]