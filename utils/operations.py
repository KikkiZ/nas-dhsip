import torch
import torch.nn as nn

UPSAMPLE_PRIMITIVE_OPS = {
    'bilinear': lambda input_channel, output_channel, kernel_size, act_op: BilinearOp(stride=2,
                                                                                      upsample_mode='bilinear',
                                                                                      act_op=act_op),
    'bicubic': lambda input_channel, output_channel, kernel_size, act_op: BilinearOp(stride=2, upsample_mode='bicubic',
                                                                                     act_op=act_op),
    'nearest': lambda input_channel, output_channel, kernel_size, act_op: BilinearOp(stride=2, upsample_mode='nearest',
                                                                                     act_op=act_op),
    'trans_conv': lambda input_channel, output_channel, kernel_size, act_op: TransConvOp(input_channel=input_channel,
                                                                                         output_channel=output_channel,
                                                                                         kernel_size=kernel_size,
                                                                                         act_op=act_op,
                                                                                         stride=2),
    'pixel_shuffle': lambda input_channel, output_channel, kernel_size, act_op: DepthToSpaceOp(act_op=act_op, stride=2),
}

UPSAMPLE_CONV_OPS = {
    'conv': lambda input_channel, output_channel, kernel_size, act_op: ConvOp(input_channel=input_channel,
                                                                              output_channel=output_channel,
                                                                              kernel_size=kernel_size,
                                                                              act_op=act_op),
    'trans_conv': lambda input_channel, output_channel, kernel_size, act_op: TransConvOp(input_channel=input_channel,
                                                                                         output_channel=output_channel,
                                                                                         kernel_size=kernel_size,
                                                                                         act_op=act_op,
                                                                                         stride=1),
    'split_stack_sum': lambda input_channel, output_channel, kernel_size, act_op: SplitStackSum(
        input_channel=input_channel, output_channel=output_channel,
        kernel_size=kernel_size, act_op=act_op),
    'sep_conv': lambda input_channel, output_channel, kernel_size, act_op: SepConvOp(input_channel=input_channel,
                                                                                     output_channel=output_channel,
                                                                                     kernel_size=kernel_size,
                                                                                     act_op=act_op),
    'depth_wise_conv': lambda input_channel, output_channel, kernel_size, act_op: DepthWiseConvOp(
        input_channel=input_channel, output_channel=output_channel,
        kernel_size=kernel_size, act_op=act_op),
    # 'identity': lambda input_channel, output_channel, kernel_size, act_op: Identity()
}

ACTIVATION_OPS = {
    'none': None,
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(0.2, inplace=False),
}


class BilinearOp(nn.Module):

    def __init__(self,
                 stride,
                 upsample_mode,
                 act_op):

        super(BilinearOp, self).__init__()

        activation = ACTIVATION_OPS[act_op]

        if not activation:
            self.op = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode=upsample_mode),
            )

        else:
            self.op = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode=upsample_mode),
                activation,
            )

    def forward(self, x):
        return self.op(x)


class DepthToSpaceOp(nn.Module):

    def __init__(self,
                 stride,
                 act_op):

        super(DepthToSpaceOp, self).__init__()

        activation = ACTIVATION_OPS[act_op]

        if activation is None:
            self.op = nn.Sequential(
                nn.PixelShuffle(stride),
            )

        else:
            self.op = nn.Sequential(
                nn.PixelShuffle(stride),
                activation,
            )

    def forward(self, x):
        return self.op(x)


class TransConvOp(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 stride,
                 act_op):

        super(TransConvOp, self).__init__()

        activation = ACTIVATION_OPS[act_op]

        if not activation:
            self.op = nn.Sequential(
                nn.ConvTranspose2d(input_channel,
                                   output_channel,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=kernel_size // 2,
                                   output_padding=stride - 1),
            )

        else:
            self.op = nn.Sequential(
                nn.ConvTranspose2d(input_channel,
                                   output_channel,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=kernel_size // 2,
                                   output_padding=stride - 1),
                activation,
            )

    def forward(self, x):
        return self.op(x)


class ConvOp(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 act_op):

        super(ConvOp, self).__init__()

        activation = ACTIVATION_OPS[act_op]

        if not activation:
            self.op = nn.Sequential(
                nn.Conv2d(input_channel,
                          output_channel,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2,
                          bias=False)
            )

        else:
            self.op = nn.Sequential(
                nn.Conv2d(input_channel,
                          output_channel,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2,
                          bias=False),
                activation
            )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SplitStackSum(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 act_op,
                 split=4):

        super(SplitStackSum, self).__init__()

        activation = ACTIVATION_OPS[act_op]

        self.chuck_size = int(input_channel / split)

        if not activation:
            self.op = nn.Sequential(
                nn.Conv2d(int(input_channel / split),
                          output_channel,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2,
                          bias=False),
            )

        else:
            self.op = nn.Sequential(
                nn.Conv2d(int(input_channel / split),
                          output_channel,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2,
                          bias=False),
                activation,
            )

    def forward(self, x):
        # the resulting number of channels will be 1/4 of the number of input channels
        split = torch.split(x, self.chuck_size, dim=1)
        # print(len(split), split[4].shape)
        stack = torch.stack(split, dim=1)
        out = torch.sum(stack, dim=1)
        out = self.op(out)
        return out


class SepConvOp(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 act_op):

        super(SepConvOp, self).__init__()

        activation = ACTIVATION_OPS[act_op]

        if not activation:
            self.op = nn.Sequential(
                nn.Conv2d(input_channel,
                          input_channel,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2,
                          groups=input_channel,
                          bias=False),
                # per channel conv
                nn.Conv2d(input_channel,
                          output_channel,
                          kernel_size=1,
                          padding=0,
                          bias=False),
                # point wise conv (1x1 conv)
            )

        else:
            self.op = nn.Sequential(
                nn.Conv2d(input_channel,
                          input_channel,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2,
                          groups=input_channel,
                          bias=False),
                # per channel conv
                nn.Conv2d(input_channel,
                          output_channel,
                          kernel_size=1,
                          padding=0,
                          bias=False),
                # point wise conv (1x1 conv)
                activation,
            )

    def forward(self, x):
        return self.op(x)


class DepthWiseConvOp(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 act_op):

        super(DepthWiseConvOp, self).__init__()

        activation = ACTIVATION_OPS[act_op]
        # print(input_channel)

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

        if activation is not None:
            self.operate.add_module(str(index), activation)

    def forward(self, x):
        # print(x.shape)
        return self.operate(x)
