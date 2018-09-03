import torch
from torch import nn


def init_conv(modules):
    """ Initializes all module weights using He initialization and set biases to zero."""
    for module in modules:
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_bn(modules):
    """ Initializes all module weights to N(1, 0.1) and set biases to zero."""
    for module in modules:
        nn.init.normal_(module.weight, mean=1, std=0.1)
        nn.init.constant_(module.bias, 0)


class DenseBlock(nn.Module):
    """ A single dense block.

    Arguments:
        in_channels (int): Number of channels in the input.
        growth_rate (int): Number of feature maps to add per layer.
        num_layers (int): Number of layers in this block.
        dilation (int): Amount of dilation in each conv layer
    """

    def __init__(self, in_channels, growth_rate, num_layers, dilation=1):
        super().__init__()
        self.out_channels = in_channels + num_layers * growth_rate
        modules = []
        for next_in_channels in range(in_channels, self.out_channels, growth_rate):
            modules.append(nn.Sequential(nn.BatchNorm3d(next_in_channels),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(next_in_channels, growth_rate, 3,
                                                   dilation=dilation, padding=dilation,
                                                   bias=False)))
        self.modules_ = nn.ModuleList(modules)

    def forward(self, input_):
        x = input_
        for module in self.modules_:
            x = torch.cat([x, module(x)], dim=1)
        return x

    def init_parameters(self):
        init_bn(module[0] for module in self.modules_)
        init_conv(module[2] for module in self.modules_)


class TransitionLayer(nn.Module):
    """ A transition layer compresses the number of feature maps.

    Arguments:
        in_channels (int): Number of channels in the input.
        compression_factor (float): How much to reduce feature maps.
    """

    def __init__(self, in_channels, compression_factor):
        super().__init__()
        self.out_channels = int(in_channels * compression_factor)
        self.layers = nn.Sequential(nn.BatchNorm3d(in_channels),
                                    nn.Conv3d(in_channels, self.out_channels, 1,
                                              bias=False))

    def forward(self, input_):
        return self.layers(input_)

    def init_parameters(self):
        init_bn([self.layers[0]])
        init_conv([self.layers[1]])


class DenseNet(nn.Module):
    """ DenseNet-C from Huang et al, 2016.

    Arguments:
        in_channels (int): Number of channels in the input images.
        initial_maps (int): Number of maps in the initial layer.
        layers_per_block (list of ints): Number of layers in each dense block. Also
            defines the number of dense blocks in the network.
        growth_rate (int): Number of feature maps to add per layer.
        compression_factor (float): Between each pair of dense blocks, the number of
            feature maps are decreased by this factor, e.g., 0.5 produces half as many.
    """
    version = 1

    def __init__(self, in_channels=1, initial_maps=16, layers_per_block=(6, 6),
                 growth_rate=8, compression_factor=0.5):
        super().__init__()

        # First conv
        self.conv1 = nn.Conv3d(in_channels, initial_maps, 3, padding=1)

        # Dense blocks and transition layers
        layers = []
        layers.append(DenseBlock(initial_maps, growth_rate, layers_per_block[0]))
        for dilation, num_layers in enumerate(layers_per_block[1:], start=2):
            layers.append(TransitionLayer(layers[-1].out_channels, compression_factor))
            layers.append(DenseBlock(layers[-1].out_channels, growth_rate, num_layers,
                                     dilation))
        self.layers = nn.Sequential(*layers)
        self.last_bias = nn.Parameter(torch.zeros(self.layers[-1].out_channels))  # *
        # * last conv does not have bias or batchnorm afterwards, so I'll add it manually

        self.out_channels = len(self.last_bias)

    def forward(self, input_):
        return self.layers(self.conv1(input_)) + self.last_bias.view(1, -1, 1, 1, 1)

    def init_parameters(self):
        nn.init.constant_(self.last_bias, 0)
        for layer in self.layers:
            layer.init_parameters()


class FCN(nn.Module):
    """ A fully convolutional network.

    Arguments:
        in_channels (int): Number of feature maps in the input.
        num_features (list): Number of feature maps per layer.
        kernel_sizes (list): Kernel size per layer.
        dilation (list): Dilation per layer.
        padding (list): Padding per layer.
    """
    version = 1

    def __init__(self, in_channels=72, num_features=(96, 1), kernel_sizes=(3, 1),
                 dilation=(1, 1), padding=(1, 0)):
        super().__init__()

        layers = []
        for ic, oc, ks, d, p in zip([in_channels, *num_features[:-1]], num_features,
                                    kernel_sizes, dilation, padding):
            layers.append(nn.Conv3d(ic, oc, ks, dilation=d, padding=p))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm3d(oc))
        layers.append(nn.Conv3d(num_features[-2], num_features[-1], kernel_sizes[-1],
                                 dilation=dilation[-1], padding=padding[-1]))  # last fc layer
        self.layers = nn.Sequential(*layers)

        self.out_channels = num_features[-1]

    def forward(self, input_):
        return self.layers(input_)

    def init_parameters(self):
        init_conv(m for m in self.layers if isinstance(m, nn.Conv3d))
        init_bn(m for m in self.layers if isinstance(m, nn.BatchNorm3d))


class QCANet(nn.Module):
    """ A two-headed network inspired by the QCANet in (Tokuoka et al., 2018).

    Image goes through an initial convolutional core that acts as a feature extractor, the
    nuclear detection network then predicts the centroid of each cell and the nuclear
    segmentation network performs cell body segmentation. Instance segmentation is later
    produced via marker-based watershed of the predicted probability heatmap with
    predicted centroids as markers.

    Arguments:
        in_channels (int): Number of channels in input image

    Returns:
        detection, segmentation: Two heatmaps of logits (same size as the input).
    """
    def __init__(self, in_channels=1):
        super().__init__()

        self.core = DenseNet(in_channels=in_channels)
        self.ndn = FCN(in_channels=self.core.out_channels)
        self.nsn = FCN(in_channels=self.core.out_channels)

    def forward(self, input_):
        hidden = self.core(input_)
        detection = self.ndn(hidden)
        segmentation = self.nsn(hidden)

        return detection, segmentation

    def init_parameters(self):
        self.core.init_parameters()
        self.ndn.init_parameters()
        self.nsn.init_parameters()