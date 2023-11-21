from torch import nn

from models import common
from utils import upsample_space


def encoder_layer(input_channel, output_channel, kernel_size, bias, pad, activate, downsample_mode):
    """ 根据参数生成一个编码器

    :param input_channel: 输入数据的通道数
    :param output_channel: 输出数据的通道数
    :param kernel_size: 卷积核的大小
    :param bias: 是否需要偏置
    :param pad: 填充类型
    :param activate: 激活函数类型
    :param downsample_mode: 下采样的类型

    :return: 返回一个nn.Sequential编码器
    """
    layer = nn.Sequential()

    # 获取特征图的同时下采样了
    layer.add_module('0:conv', common.conv(input_channel=input_channel,
                                           output_channel=output_channel,
                                           downsample_mode=downsample_mode,
                                           kernel_size=kernel_size,
                                           stride=2,
                                           bias=bias,
                                           pad=pad))
    layer.add_module('1:bn', common.bn(output_channel))
    layer.add_module('2:act', common.act(activate=activate))
    layer.add_module('3:conv', common.conv(input_channel=output_channel,
                                           output_channel=output_channel,
                                           kernel_size=kernel_size,
                                           bias=bias,
                                           pad=pad))
    layer.add_module('4:bn', common.bn(output_channel))
    layer.add_module('5:act', common.act(activate=activate))

    return layer


def decoder_layer(input_channel, output_channel, kernel_size, bias, pad, activate, need1x1_up):
    """ 根据参数生成一个解码器

    :param input_channel: 输入数据的通道数
    :param output_channel: 输出数据的通道数
    :param kernel_size: 卷积核的大小
    :param bias: 是否需要偏置
    :param pad: 填充类型
    :param activate: 激活函数类型
    :param need1x1_up: 是否需要1x1卷积

    :return: 返回一个nn.Sequential解码器
    """
    layer = nn.Sequential()

    layer.add_module('0:bn', common.bn(num_features=input_channel))
    layer.add_module('1:conv', common.conv(input_channel=input_channel,
                                           output_channel=output_channel,
                                           kernel_size=kernel_size,
                                           bias=bias,
                                           pad=pad))
    layer.add_module('2:bn', common.bn(num_features=output_channel))
    layer.add_module('3:act', common.act(activate=activate))
    layer.add_module('4:conv', common.conv(input_channel=output_channel,
                                           output_channel=output_channel,
                                           kernel_size=kernel_size,
                                           bias=bias,
                                           pad=pad))
    layer.add_module('5:bn', common.bn(num_features=output_channel))
    layer.add_module('6:act', common.act(activate=activate))

    if need1x1_up:
        layer.add_module('7:conv', common.conv(input_channel=output_channel,
                                               output_channel=output_channel,
                                               kernel_size=1,
                                               bias=bias,
                                               pad=pad))
        layer.add_module('8:bn', common.bn(num_features=output_channel))
        layer.add_module('9:act', common.act(activate=activate))

    return layer


def downsample_layer(input_channel, output_channel, kernel_size, bias, pad, activate, downsample_mode):
    """ 根据参数生成一个下采样层

    :param input_channel: 输入数据的通道数
    :param output_channel: 输出数据的通道数
    :param kernel_size: 卷积核的大小
    :param bias: 是否需要偏置
    :param pad: 填充类型
    :param activate: 激活函数类型
    :param downsample_mode: 下采样的类型

    :return: 返回一个nn.Sequential下采样层
    """
    layer = nn.Sequential()

    layer.add_module('0:conv', common.conv(input_channel=input_channel,
                                           output_channel=output_channel,
                                           downsample_mode=downsample_mode,
                                           kernel_size=kernel_size,
                                           stride=2,
                                           bias=bias,
                                           pad=pad))
    layer.add_module('1:bn', common.bn(num_features=output_channel))
    layer.add_module('2:act', common.act(activate=activate))

    return layer


def upsample_layer(input_channel, output_channel, upsample_params: upsample_space.Upsample):
    """ 根据RNN controller生成的参数生成一个上采样层

    :param input_channel: 输入数据的通道数
    :param output_channel: 输出结果的通道数
    :param upsample_params: NAS生成的参数
    :return: 返回一个nn.Sequential上采样层
    """
    layer = nn.Sequential()

    # 解析RNN controller生成的参数
    primitive_operate = upsample_space.UPSAMPLE_PRIMITIVE[upsample_params.primitive]
    conv_operate = upsample_space.UPSAMPLE_CONV[upsample_params.conv]
    kernel_size = upsample_space.KERNEL_SIZE[upsample_params.kernel]
    activation_operate = upsample_space.ACTIVATION[upsample_params.activation]

    # 根据参数生成对应的上采样层
    primitive_layer = upsample_space.UPSAMPLE_PRIMITIVE_OPTIONS[primitive_operate](input_channel=input_channel,
                                                                                   output_channel=output_channel,
                                                                                   kernel_size=kernel_size,
                                                                                   act_op=activation_operate)

    # 生成对应的卷积层
    if primitive_operate == 'pixel_shuffle':
        conv_layer = upsample_space.UPSAMPLE_CONV_OPTIONS[conv_operate](input_channel=input_channel // 4,
                                                                        output_channel=output_channel,
                                                                        kernel_size=kernel_size,
                                                                        act_op=activation_operate)
    elif primitive_operate == 'trans_conv':
        conv_layer = upsample_space.UPSAMPLE_CONV_OPTIONS[conv_operate](input_channel=output_channel,
                                                                        output_channel=output_channel,
                                                                        kernel_size=kernel_size,
                                                                        act_op=activation_operate)
    else:
        conv_layer = upsample_space.UPSAMPLE_CONV_OPTIONS[conv_operate](input_channel=input_channel,
                                                                        output_channel=output_channel,
                                                                        kernel_size=kernel_size,
                                                                        act_op=activation_operate)

    layer.add_module('0:primitive', primitive_layer)
    layer.add_module('1:conv', conv_layer)

    return layer


def skip_layer(input_channel, output_channel, kernel_size, bias, pad, activate):
    """ 根据参数生成一个跳跃连接

    :param input_channel: 输入数据的通道数
    :param output_channel: 输出数据的通道数
    :param kernel_size: 卷积核的大小
    :param bias: 是否需要偏置
    :param pad: 填充类型
    :param activate: 激活函数类型

    :return: 返回一个nn.Sequential跳跃连接层
    """
    layer = nn.Sequential()

    layer.add_module('0:conv', common.conv(input_channel=input_channel,
                                           output_channel=output_channel,
                                           kernel_size=kernel_size,
                                           bias=bias,
                                           pad=pad))
    layer.add_module('1:bn', common.bn(num_features=output_channel))
    layer.add_module('2:act', common.act(activate=activate))

    return layer


def output_layer(input_channel, output_channel, bias, pad, need_sigmoid):
    """ 根据参数生成网络的输出层

    :param input_channel: 输入数据的通道数
    :param output_channel: 输出数据的通道数
    :param bias: 是否需要偏置
    :param pad: 填充类型
    :param need_sigmoid: 在输出前是否需要激活函数

    :return: 返回一个nn.Sequential输出层
    """
    layer = nn.Sequential()

    layer.add_module('0:conv', common.conv(input_channel=input_channel,
                                           output_channel=output_channel,
                                           kernel_size=1,
                                           bias=bias,
                                           pad=pad))

    if need_sigmoid:
        layer.add_module('1:act', nn.Sigmoid())

    return layer
