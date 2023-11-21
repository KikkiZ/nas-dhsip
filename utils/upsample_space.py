from collections import namedtuple

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
