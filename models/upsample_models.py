import torch
from torch import nn

from utils import upsample_space


class BilinearModule(nn.Module):

    def __init__(self, stride, upsample_mode, activation):
        super(BilinearModule, self).__init__()

        upsample = nn.Upsample(scale_factor=stride,
                               mode=upsample_mode)
        self.operate = nn.Sequential()
        self.operate.add_module('0', upsample)

        activation = upsample_space.ACTIVATION_OPTIONS[activation]
        if activation is not None:
            self.operate.add_module('1', activation)

    def forward(self, x):
        return self.operate(x)


class DepthToSpaceModule(nn.Module):

    def __init__(self, stride, activation):
        super(DepthToSpaceModule, self).__init__()

        activation = upsample_space.ACTIVATION_OPTIONS[activation]

        if activation is None:
            self.operate = nn.Sequential(
                nn.PixelShuffle(stride),
            )
        else:
            self.operate = nn.Sequential(
                nn.PixelShuffle(stride),
                activation,
            )

    def forward(self, x):
        return self.operate(x)


class TransConvModule(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, stride, activation):
        super(TransConvModule, self).__init__()

        conv = nn.ConvTranspose2d(input_channel,
                                  output_channel,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=kernel_size // 2,
                                  output_padding=stride - 1)
        self.operate = nn.Sequential()
        self.operate.add_module('0', conv)

        activation = upsample_space.ACTIVATION_OPTIONS[activation]
        if activation is None:
            self.operate.add_module('1', activation)

    def forward(self, x):
        return self.operate(x)


class ConvModule(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, activation):
        super(ConvModule, self).__init__()

        conv = nn.Conv2d(input_channel,
                         output_channel,
                         kernel_size=kernel_size,
                         padding=kernel_size // 2,
                         bias=False)
        self.operate = nn.Sequential()
        self.operate.add_module('0', conv)

        activation = upsample_space.ACTIVATION_OPTIONS[activation]
        if activation is None:
            self.operate.add_module('1', activation)

    def forward(self, x):
        return self.operate(x)


# class Identity(nn.Module):
#
#     def __init__(self):
#         super(Identity, self).__init__()
#
#     def forward(self, x):
#         return x


class SplitStackModule(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, activation, split=4):
        super(SplitStackModule, self).__init__()

        conv = nn.Conv2d(input_channel // split,
                         output_channel,
                         kernel_size=kernel_size,
                         padding=kernel_size // 2,
                         bias=False)
        self.operate = nn.Sequential()
        self.operate.add_module('0', conv)

        activation = upsample_space.ACTIVATION_OPTIONS[activation]
        if activation is None:
            self.operate.add_module('1', activation)

    def forward(self, x):
        # the resulting number of channels will be 1/4 of the number of input channels
        split = torch.split(x, self.chuck_size, dim=1)
        stack = torch.stack(split, dim=1)
        out = torch.sum(stack, dim=1)
        out = self.operate(out)
        return out


class SepConvModule(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, activation):
        super(SepConvModule, self).__init__()

        conv_1 = nn.Conv2d(input_channel,
                           input_channel,
                           kernel_size=kernel_size,
                           padding=kernel_size // 2,
                           groups=input_channel,
                           bias=False)
        conv_2 = nn.Conv2d(input_channel,
                           output_channel,
                           kernel_size=1,
                           padding=0,
                           bias=False)
        self.operate = nn.Sequential()
        self.operate.add_module('0', conv_1)
        self.operate.add_module('1', conv_2)

        activation = upsample_space.ACTIVATION_OPTIONS[activation]
        if activation is None:
            self.operate.add_module('2', activation)

    def forward(self, x):
        return self.operate(x)


class DepthWiseConvModule(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, activation):
        super(DepthWiseConvModule, self).__init__()

        upsample = nn.Conv2d(input_channel,
                             input_channel,
                             kernel_size=kernel_size,
                             padding=kernel_size // 2,
                             groups=input_channel,
                             bias=False)
        self.operate = nn.Sequential()
        self.operate.add_module('0', module=upsample)

        index = 1
        if input_channel != output_channel:
            self.operate.add_module(str(index), nn.Conv2d(input_channel, output_channel, kernel_size=1))
            index = 2

        activation = upsample_space.ACTIVATION_OPTIONS[activation]
        if activation is not None:
            self.operate.add_module(str(index), activation)

    def forward(self, x):
        return self.operate(x)
