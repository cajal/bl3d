""" PyTorch definition of different models as nn.Modules. """
from torch import nn
from torch.nn import functional as F


def init_conv(modules):
    """ Initializes all module weights using He initialization and set biases to zero."""
    for module in modules:
        nn.init.kaiming_normal(module.weight)
        nn.init.constant(module.bias, 0)

def arr2int(array):
    """ From array of numpy integers to tuple of Python ints."""
    return tuple(int(x) for x in array)


class LinearFilter(nn.Module):
    """ A single learnable 3-d filter.

    Arguments:
        num_features: Tuple. Number of channels in the input and output volumes.
        filter_size: Filter size sent to Conv3d.
    """
    def __init__(self, num_features=(1, 2), filter_size=(25, 19, 19)):
        super().__init__()
        self.filter = nn.Conv3d(int(num_features[0]), int(num_features[1]),
                                arr2int(filter_size), padding=arr2int(filter_size / 2))

    def forward(self, input_):
        output = self.filter(input_)
        return output

    def init_parameters(self):
        init_conv([self.filter])


class Dictionary(nn.Module):
    """ A two-layer convnet with 3-d filters learned in the first layer and a fully
        connected layer on top of it.

    Arguments:
        num_features: Triple. Number of features per layer (input, hidden and output layer).
        filter_size: Filter size sent to Conv3d.
    """
    def __init__(self, num_features=(1, 16, 2), filter_size=(25, 19, 19), use_batchnorm=False):
        super().__init__()
        self.conv = nn.Conv3d(int(num_features[0]), int(num_features[1]),
                                  arr2int(filter_size), padding=arr2int(filter_size / 2))
        self.bn = nn.BatchNorm3d(int(num_features[1])) if use_batchnorm else nn.Sequential() # bn or no-op
        self.fc = nn.Conv3d(int(num_features[1]), int(num_features[2]), 1)

    def forward(self, input_):
        h1 = self.bn(F.relu(self.conv(input_), inplace=True))
        output = self.fc(h1)
        return output

    def init_parameters(self):
        init_conv([self.conv, self.fc])


class FullyConvNet(nn.Module):
    """ A fully convolutional network.

    Default params have 7 layers and an effective receptive field of 19 x 19.
    """
    def __init__(self, num_features=(1, 8, 8, 16, 16, 32, 32, 2), kernel_sizes=(3, 3, 3, 3, 3, 1, 1),
                 dilation=(1, 1, 2, 2, 3, 1, 1), padding=(1, 1, 2, 2, 3, 0, 0), use_batchnorm=False):
        super().__init__()

        modules = []
        for i in range(len(num_features) - 2): # each conv layer
            modules.append(nn.Conv3d(int(num_features[i]), int(num_features[i + 1]),
                                     int(kernel_sizes[i]), dilation=int(dilation[i]),
                                     padding=int(padding[i])))
            modules.append(nn.ReLU(inplace=True))
            if use_batchnorm:
                modules.append(nn.BatchNorm3d(int(num_features[i])))
        modules.append(nn.Conv3d(int(num_features[-2]), int(num_features[-1]),
                                 int(kernel_sizes[-1]), dilation=int(dilation[-1]),
                                 padding=int(padding[-1]))) # last fc layer
        self.fcn = nn.Sequential(*modules)

    def forward(self, input_):
        output = self.fcn(input_)
        return output

    def init_parameters(self):
        init_conv([module for module in self.fcn if isinstance(module, nn.Conv3d)])

#TODO: Mask R-CNN (https://arxiv.org/abs/1703.06870)