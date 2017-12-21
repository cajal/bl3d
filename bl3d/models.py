""" PyTorch definition of different models as nn.Modules. """
from torch import nn
from torch.nn import functional as F


def init_conv(modules):
    """ Initializes all module weights using He initialization and set biases to zero."""
    for module in modules:
        nn.init.kaiming_normal(module.weight)
        nn.init.constant(module.bias, 0)


class LinearFilter(nn.Module):
    """ A single 31 x 31 learnable filter.

    Arguments:
        in_channels: Number of channels in the input volume.
        out_channels: Number of channels of the required output. Usually the number of
            classes.
    """
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.filter = nn.Conv3d(in_channels, out_channels, (25, 19, 19), padding=(12, 9, 9))

    def forward(self, input_):
        output = self.filter(input_)
        return output

    def init_params(self):
        init_conv([self.filter])


class Dictionary(nn.Module):
    """ A two-layer convnet with 16 3-d filters learned in the first layer and a fully
        connected layer in top of it.

    Arguments:
        in_channels: Number of channels in the input volume.
        out_channels: Number of channels of the required output. Usually the number of
            classes.

    Effective filter size: 25 x 18 x 18
    """
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 16, (25, 19, 19), padding=(12, 9, 9))
        self.fc = nn.Conv3d(16, out_channels, 1)

    def forward(self, input_):
        h1 = F.relu(self.conv(input_))
        output = self.fc(h1)
        return output

    def init_params(self):
        init_conv([self.conv, self.fc])


class FullyConvNet(nn.Module):
    """ A 7 layer convnet.

    Effective filter size: 19 x 19 x 19
    """
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 8, 3, padding=1) # Change this to 5
        self.conv2 = nn.Conv3d(8, 8, 3, padding=1)
        self.conv3 = nn.Conv3d(8, 16, 3, padding=2, dilation=2)
        self.conv4 = nn.Conv3d(16, 16, 3, padding=2, dilation=2)
        self.conv5 = nn.Conv3d(16, 32, 3, padding=3, dilation=3)
        self.fc1 = nn.Conv3d(32, 32, 1)
        self.fc2 = nn.Conv3d(32, out_channels, 1)

    def forward(self, input_):
        h1 = F.relu(self.conv1(input_))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h5 = F.relu(self.conv5(h4))
        h6 = self.fc1(h5)
        output = self.fc2(h6)
        return output

    def init_params(self):
        init_conv([module for module in self.modules if isinstance(module, nn.Conv3d)])


#class FullyConvNet_plus_BN(nn.Module):
#    #TODO: CONV-> RELU-> BN
