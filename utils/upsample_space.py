from collections import namedtuple

from torch import nn

from models import upsample_models as models

Upsample = namedtuple('Upsample',
                      'primitive conv kernel activation')

UPSAMPLE_PRIMITIVE = [
    'bilinear',
    'bicubic',
    'nearest',
    'trans_conv',
    'pixel_shuffle'
]

UPSAMPLE_CONV = [
    'conv',
    'trans_conv',
    'split_stack_sum',
    'sep_conv',
    'depth_wise_conv',
    # 'identity'
]

KERNEL_SIZE = [
    3,
    5,
    7
]

ACTIVATION = [
    'none',
    'ReLU',
    'LeakyReLU'
]

UPSAMPLE_PRIMITIVE_OPTIONS = {
    'bilinear': lambda input_channel, output_channel, kernel_size, act_op:
    models.BilinearModule(stride=2,
                          upsample_mode='bilinear',
                          activation=act_op),
    'bicubic': lambda input_channel, output_channel, kernel_size, act_op:
    models.BilinearModule(stride=2,
                          upsample_mode='bicubic',
                          activation=act_op),
    'nearest': lambda input_channel, output_channel, kernel_size, act_op:
    models.BilinearModule(stride=2,
                          upsample_mode='nearest',
                          activation=act_op),
    'trans_conv': lambda input_channel, output_channel, kernel_size, act_op:
    models.TransConvModule(input_channel=input_channel,
                           output_channel=output_channel,
                           kernel_size=kernel_size,
                           activation=act_op,
                           stride=2),
    'pixel_shuffle': lambda input_channel, output_channel, kernel_size, act_op:
    models.DepthToSpaceModule(activation=act_op, stride=2)
}

UPSAMPLE_CONV_OPTIONS = {
    'conv': lambda input_channel, output_channel, kernel_size, act_op:
    models.ConvModule(input_channel=input_channel,
                      output_channel=output_channel,
                      kernel_size=kernel_size,
                      activation=act_op),
    'trans_conv': lambda input_channel, output_channel, kernel_size, act_op:
    models.TransConvModule(input_channel=input_channel,
                           output_channel=output_channel,
                           kernel_size=kernel_size,
                           activation=act_op,
                           stride=1),
    'split_stack_sum': lambda input_channel, output_channel, kernel_size, act_op:
    models.SplitStackModule(input_channel=input_channel,
                            output_channel=output_channel,
                            kernel_size=kernel_size,
                            activation=act_op),
    'sep_conv': lambda input_channel, output_channel, kernel_size, act_op:
    models.SepConvModule(input_channel=input_channel,
                         output_channel=output_channel,
                         kernel_size=kernel_size,
                         activation=act_op),
    'depth_wise_conv': lambda input_channel, output_channel, kernel_size, act_op:
    models.DepthWiseConvModule(input_channel=input_channel,
                               output_channel=output_channel,
                               kernel_size=kernel_size,
                               activation=act_op)
}

ACTIVATION_OPTIONS = {
    'none': None,
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(0.2, inplace=False),
}
