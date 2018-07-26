""" Hyperparameter configurations. """
import datajoint as dj


schema = dj.schema('ecobost_bl3d2', locals())


@schema
class TrainingParams(dj.Lookup):
    definition = """ # different hyperparameters to search over

    training_id:        int             # unique id for hyperparameter combination
    ---
    seed:               int             # random seed for torch/np
    train_crop_size:    int             # size of the random crops used for training
    val_crop_size:      int             # size of the random crops used for validation
    enhanced_input:     boolean         # whether to use enhanced input
    anchor_size_d:      tinyint         # size of the anchor (depth)
    anchor_size_w:      tinyint         # size of the anchor (width)
    anchor_size_h:      tinyint         # size of the anchor (height)

    train_num_proposals:int             # number of ROI proposals to refine and segment during training
    val_num_proposals:  int             # number of ROI proposals to refine and segment during validation
    nms_iou:            float           # IOU used as non-maximum-suppression threshold
    roi_size_d:         tinyint         # size of the extracted ROIs (depth)
    roi_size_h:         tinyint         # size of the extracted ROIs (height)
    roi_size_w:         tinyint         # size of the extracted ROIs (width)

    learning_rate:      float           # initial learning rate for SGD
    weight_decay:       float           # lambda for l2-norm weight regularization
    momentum:           float           # momentum factor for SGD updates
    num_epochs:         int             # number of training epochs
    lr_decay:           float           # factor multiplying learning rate when decreasing
    lr_schedule:        tinyblob        # epochs after which learning rate will be decreased
    positive_weight:    float           # relative weight for RPN positive class examples (negative class weight is 1)
    smoothl1_weight:    float           # weight given to the smooth-l1 loss component of the RPN and bbox loss
    """
    @property
    def contents(self):
        import itertools
        search_params = itertools.product([0.001, 0.01, 0.1], [1e-5, 1e-4, 1e-3], [1, 10])
        for id_, (lr, lambda_, sl1_weight) in enumerate(search_params, start=1):
            yield {'training_id': id_, 'learning_rate': lr, 'weight_decay': lambda_,
                   'smoothl1_weight': sl1_weight, 'seed': 1234, 'train_crop_size': 128,
                   'val_crop_size': 192, 'enhanced_input': False, 'anchor_size_d': 15,
                   'anchor_size_w': 9, 'anchor_size_h': 9, 'train_num_proposals': 1024,
                   'val_num_proposals': 2048, 'nms_iou': 0.25, 'roi_size_d': 12,
                   'roi_size_h': 12, 'roi_size_w': 12, 'momentum': 0.9, 'num_epochs': 140,
                   'lr_decay': 0.1, 'lr_schedule': (100, 130), 'positive_weight': 5}


@schema
class ModelParams(dj.Lookup):
    definition = """ # architectural details of our different models

    model_version:      tinyint         # unique id for this network
    ---
    -> DenseNet
    -> RPN
    -> Bbox
    -> FCN
    """
    contents = [
        [1, 1, 1, 1, 1]
    ]


@schema
class DenseNet(dj.Lookup):
    definition = """

    core_version:       tinyint
    ---
    num_layers:         tinyint         # number of layers
    growth_rate:        tinyint         # how many feature maps to add each layer
    kernel_sizes:       tinyblob        # list with kernel sizes (one per layer)
    dilation:           tinyblob        # list with dilation (one per layer)
    stride:             tinyblob        # list with strides (one per layer)
    use_batchnorm:      boolean         # whether to use batch normalization
    """
    contents = [
        [1, 5, 8, (3, 3, 3, 3, 3), (1, 1, 2, 2, 3), (1, 1, 1, 1, 1), True]
    ]


@schema
class RPN(dj.Lookup):
    definition = """

    rpn_version:        tinyint
    ---
    num_features:       tinyblob        # number of feature maps per layer
    kernel_sizes:       tinyblob        # list with kernel sizes (one per layer)
    dilation:           tinyblob        # list with dilation (one per layer)
    stride:             tinyblob        # list with strides (one per layer)
    use_batchnorm:      boolean         # whether to use batch normalization
    """
    contents = [
        [1, (48, 64, 7), (3, 1, 1), (3, 1, 1), (1, 1, 1), True]
    ]


@schema
class Bbox(dj.Lookup):
    definition = """

    bbox_version:       tinyint
    ---
    num_features:       tinyblob        # number of feature maps per layer
    avg_pool:           tinyint         # average pool after this layer
    kernel_sizes:       tinyblob        # list with kernel sizes (one per conv layer)
    dilation:           tinyblob        # list with dilation (one per conv layer)
    stride:             tinyblob        # list with strides (one per conv layer)
    use_batchnorm:      boolean         # whether to use batch normalization
    """
    contents = [
        [1, (48, 48, 48, 96, 7), 3, (3, 3, 3), (1, 1, 1), (1, 2, 1), True]
    ]


@schema
class FCN(dj.Lookup):
    definition = """

    fcn_version:        tinyint
    ---
    num_features:       tinyblob        # number of feature maps per layer
    kernel_sizes:       tinyblob        # list with kernel sizes (one per layer)
    dilation:           tinyblob        # list with dilation (one per layer)
    stride:             tinyblob        # list with strides (one per layer)
    use_batchnorm:      boolean         # whether to use batch normalization
    """
    contents = [
        [1, (48, 48, 48, 96, 1), (3, 3, 3, 1, 1), (1, 2, 2, 1, 1), (1, 1, 1, 1, 1), True]
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


@schema
class EvalParams(dj.Lookup):
    definition=""" # parameters used during evaluation

    eval_id:            int         # id for the evaluation parameters
    ---
    nms_iou:            float       # IOU used as non-maximum suppression threshold
    cells_per_um :      float       # estimated number of cells per 1 um cube
    proposal_factor:    float       # estimated cells * proposal_factor = num_proposals
    mask_factor:        float       # estimated cells * masks_factor = num_masks
    mask_threshold:     float       # probability threshold for mask binarization
    """
    contents = [
        [1, 0.5, 0.0001, 4, 1.5, 0.5]
    ]


@schema
class EvalSet(dj.Lookup):
    definition = """ # set where metrics are computed
    eval_set:    varchar(8)
    """
    contents = [
        ['train'],
        ['val']
    ]