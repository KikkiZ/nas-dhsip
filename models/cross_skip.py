import upsample_layer

from common import *


class OutputBlock(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 bias,
                 pad,
                 need_sigmoid):

        super(OutputBlock, self).__init__()

        if need_sigmoid:
            self.operate = nn.Sequential(conv(in_f=input_channel,
                                              out_f=output_channel,
                                              kernel_size=kernel_size,
                                              bias=bias,
                                              pad=pad),
                                         nn.Sigmoid())
        else:
            self.operate = nn.Sequential(
                conv(in_f=input_channel,
                     out_f=output_channel,
                     kernel_size=kernel_size,
                     bias=bias,
                     pad=pad)
            )

    def forward(self, data):
        return self.operate(data)


class UpsampleBlock(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 model_index):
        super(UpsampleBlock, self).__init__()

        self.operate = upsample_layer.generate_layer(
            input_channel=input_channel,
            output_channel=output_channel,
            model_index=model_index
        )

    def forward(self, data):
        return self.operate(data)


class DownsampleBlock(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 bias,
                 pad,
                 act_fun,
                 downsample_mode):
        super(DownsampleBlock, self).__init__()

        self.operate = nn.Sequential(
            conv(input_channel=input_channel,
                 output_channel=output_channel,
                 kernel_size=kernel_size,
                 stride=2,
                 bias=bias,
                 pad=pad,
                 downsample_mode=downsample_mode),
            bn(num_features=output_channel),
            act(activate=act_fun)
        )

    def forward(self, data):
        return self.operate(data)


class SkipBlock(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 bias,
                 pad,
                 act_fun):
        super(SkipBlock, self).__init__()

        self.operate = nn.Sequential(
            conv(in_f=input_channel,
                 out_f=output_channel,
                 kernel_size=kernel_size,
                 bias=bias,
                 pad=pad),
            bn(num_features=output_channel),
            act(act_fun=act_fun)
        )

    def forward(self, data):
        return self.operate(data)


class EncoderBlock(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 bias,
                 pad,
                 act_fun,
                 downsample_mode):
        super(EncoderBlock, self).__init__()

        self.operate = nn.Sequential(
            conv(in_f=input_channel,
                 out_f=output_channel,
                 kernel_size=kernel_size,
                 stride=2,
                 bias=bias,
                 pad=pad,
                 downsample_mode=downsample_mode),
            bn(num_features=output_channel),
            act(act_fun=act_fun),
            conv(in_f=output_channel,
                 out_f=output_channel,
                 kernel_size=kernel_size,
                 bias=bias,
                 pad=pad),
            bn(num_features=output_channel),
            act(act_fun=act_fun),
        )

    def forward(self, data):
        return self.operate(data)


class DecoderBlock(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 bias,
                 pad,
                 act_fun,
                 need1x1_up):

        super(DecoderBlock, self).__init__()

        if need1x1_up:
            self.operate = nn.Sequential(
                bn(num_features=output_channel),
                conv(in_f=input_channel,
                     out_f=output_channel,
                     kernel_size=kernel_size,
                     stride=1, bias=bias, pad=pad),
                bn(num_features=output_channel),
                act(act_fun=act_fun),
                conv(in_f=output_channel,
                     out_f=output_channel,
                     kernel_size=kernel_size,
                     bias=bias, pad=pad),
                bn(num_features=output_channel),
                act(act_fun=act_fun),
                conv(in_f=output_channel,
                     out_f=output_channel,
                     kernel_size=1, bias=bias, pad=pad),
                bn(num_features=output_channel),
                act(act_fun=act_fun),
            )

        else:
            self.operate = nn.Sequential(
                bn(num_features=output_channel),
                conv(in_f=input_channel, out_f=output_channel, kernel_size=kernel_size, stride=1, bias=bias, pad=pad),
                bn(num_features=output_channel),
                act(act_fun=act_fun),
                conv(in_f=output_channel, out_f=output_channel, kernel_size=kernel_size, bias=bias, pad=pad),
                bn(num_features=output_channel),
                act(act_fun=act_fun),
            )

    def forward(self, data):
        return self.operate(data)


class Skip(nn.Module):

    def __init__(self,
                 model_index,
                 skip_index,
                 num_input_channels=2,
                 num_output_channels=3,
                 num_channels_down=None,
                 num_channels_up=None,
                 filter_size_down=3,
                 filter_size_up=3,
                 need_sigmoid=True,
                 need_bias=True,
                 pad='zero',
                 downsample_mode='stride',
                 act_fun='LeakyReLU',
                 need1x1_up=True):

        super(Skip, self).__init__()

        if num_channels_up is None:
            num_channels_up = [16, 32, 64, 128, 128]
        if num_channels_down is None:
            num_channels_down = [16, 32, 64, 128, 128]
        self.skip_index = skip_index

        """ Encoder block """
        self.enc1 = EncoderBlock(input_channel=num_input_channels,
                                 output_channel=num_channels_down[0],
                                 kernel_size=filter_size_down,
                                 bias=need_bias,
                                 pad=pad,
                                 act_fun=act_fun,
                                 downsample_mode=downsample_mode)
        self.enc2 = EncoderBlock(input_channel=num_channels_down[0],
                                 output_channel=num_channels_down[1],
                                 kernel_size=filter_size_down,
                                 bias=need_bias,
                                 pad=pad,
                                 act_fun=act_fun,
                                 downsample_mode=downsample_mode)
        self.enc3 = EncoderBlock(input_channel=num_channels_down[1],
                                 output_channel=num_channels_down[2],
                                 kernel_size=filter_size_down,
                                 bias=need_bias,
                                 pad=pad,
                                 act_fun=act_fun,
                                 downsample_mode=downsample_mode)
        self.enc4 = EncoderBlock(input_channel=num_channels_down[2],
                                 output_channel=num_channels_down[3],
                                 kernel_size=filter_size_down,
                                 bias=need_bias,
                                 pad=pad,
                                 act_fun=act_fun,
                                 downsample_mode=downsample_mode)
        self.enc5 = EncoderBlock(input_channel=num_channels_down[3],
                                 output_channel=num_channels_down[4],
                                 kernel_size=filter_size_down,
                                 bias=need_bias,
                                 pad=pad,
                                 act_fun=act_fun,
                                 downsample_mode=downsample_mode)

        """ Same-scale (level) skip connections """
        self.skip1 = SkipBlock(input_channel=num_input_channels,
                               output_channel=num_channels_up[0],
                               kernel_size=1,
                               bias=need_bias,
                               pad=pad,
                               act_fun=act_fun)
        self.skip2 = SkipBlock(input_channel=num_channels_down[0],
                               output_channel=num_channels_up[1],
                               kernel_size=1,
                               bias=need_bias,
                               pad=pad,
                               act_fun=act_fun)
        self.skip3 = SkipBlock(input_channel=num_channels_down[1],
                               output_channel=num_channels_up[2],
                               kernel_size=1,
                               bias=need_bias,
                               pad=pad,
                               act_fun=act_fun)
        self.skip4 = SkipBlock(input_channel=num_channels_down[2],
                               output_channel=num_channels_up[3],
                               kernel_size=1,
                               bias=need_bias,
                               pad=pad,
                               act_fun=act_fun)
        self.skip5 = SkipBlock(input_channel=num_channels_down[3],
                               output_channel=num_channels_up[4],
                               kernel_size=1,
                               bias=need_bias,
                               pad=pad,
                               act_fun=act_fun)

        """ Cross-scale upsample skip connections - shared in the same feature level """
        self.skip_up_5_4 = UpsampleBlock(input_channel=num_channels_down[4],
                                         output_channel=num_channels_up[3],
                                         model_index=model_index)
        self.skip_up_4_3 = UpsampleBlock(input_channel=num_channels_down[3],
                                         output_channel=num_channels_up[2],
                                         model_index=model_index)
        self.skip_up_3_2 = UpsampleBlock(input_channel=num_channels_down[2],
                                         output_channel=num_channels_up[1],
                                         model_index=model_index)
        self.skip_up_2_1 = UpsampleBlock(input_channel=num_channels_down[1],
                                         output_channel=num_channels_up[0],
                                         model_index=model_index)

        """ Cross-scale downsample skip connections - shared in the same feature level """
        self.skip_down_1_2 = DownsampleBlock(input_channel=num_input_channels,
                                             output_channel=num_channels_up[0],
                                             kernel_size=filter_size_down,
                                             bias=need_bias,
                                             pad=pad,
                                             act_fun=act_fun,
                                             downsample_mode=downsample_mode)
        self.skip_down_2_3 = DownsampleBlock(input_channel=num_channels_down[0],
                                             output_channel=num_channels_up[1],
                                             kernel_size=filter_size_down,
                                             bias=need_bias,
                                             pad=pad,
                                             act_fun=act_fun,
                                             downsample_mode=downsample_mode)
        self.skip_down_3_4 = DownsampleBlock(input_channel=num_channels_down[1],
                                             output_channel=num_channels_up[2],
                                             kernel_size=filter_size_down,
                                             bias=need_bias,
                                             pad=pad,
                                             act_fun=act_fun,
                                             downsample_mode=downsample_mode)
        self.skip_down_4_5 = DownsampleBlock(input_channel=num_channels_down[2],
                                             output_channel=num_channels_up[3],
                                             kernel_size=filter_size_down,
                                             bias=need_bias,
                                             pad=pad,
                                             act_fun=act_fun,
                                             downsample_mode=downsample_mode)

        """ Upsampling layers in the decoder  """
        self.up5 = UpsampleBlock(input_channel=num_channels_up[4],
                                 output_channel=num_channels_up[4],
                                 model_index=model_index)
        self.up4 = UpsampleBlock(input_channel=num_channels_up[3],
                                 output_channel=num_channels_up[3],
                                 model_index=model_index)
        self.up3 = UpsampleBlock(input_channel=num_channels_up[2],
                                 output_channel=num_channels_up[2],
                                 model_index=model_index)
        self.up2 = UpsampleBlock(input_channel=num_channels_up[1],
                                 output_channel=num_channels_up[1],
                                 model_index=model_index)
        self.up1 = UpsampleBlock(input_channel=num_channels_up[0],
                                 output_channel=num_channels_up[0],
                                 model_index=model_index)

        """ Decoder block """
        self.dec5 = DecoderBlock(input_channel=num_channels_down[4],
                                 output_channel=num_channels_up[4],
                                 kernel_size=filter_size_up,
                                 bias=need_bias,
                                 pad=pad,
                                 act_fun=act_fun,
                                 need1x1_up=need1x1_up)
        self.dec4 = DecoderBlock(input_channel=num_channels_up[3],
                                 output_channel=num_channels_up[3],
                                 kernel_size=filter_size_up,
                                 bias=need_bias,
                                 pad=pad,
                                 act_fun=act_fun,
                                 need1x1_up=need1x1_up)
        self.dec3 = DecoderBlock(input_channel=num_channels_up[2],
                                 output_channel=num_channels_up[2],
                                 kernel_size=filter_size_up,
                                 bias=need_bias,
                                 pad=pad,
                                 act_fun=act_fun,
                                 need1x1_up=need1x1_up)
        self.dec2 = DecoderBlock(input_channel=num_channels_up[1],
                                 output_channel=num_channels_up[1],
                                 kernel_size=filter_size_up,
                                 bias=need_bias,
                                 pad=pad,
                                 act_fun=act_fun,
                                 need1x1_up=need1x1_up)
        self.dec1 = DecoderBlock(input_channel=num_channels_up[0],
                                 output_channel=num_channels_up[0],
                                 kernel_size=filter_size_up,
                                 bias=need_bias,
                                 pad=pad,
                                 act_fun=act_fun,
                                 need1x1_up=need1x1_up)

        """ Output block """
        self.output = OutputBlock(input_channel=num_channels_up[0],
                                  output_channel=num_output_channels,
                                  kernel_size=1,
                                  bias=need_bias,
                                  pad=pad,
                                  need_sigmoid=need_sigmoid)

    def forward(self, data):

        """ Encoder """
        enc1 = self.enc1(data)  # H/2  x W/2  x 128
        enc2 = self.enc2(enc1)  # H/4  x W/4  x 128
        enc3 = self.enc3(enc2)  # H/8  x W/8  x 128
        enc4 = self.enc4(enc3)  # H/16 x W/16 x 128
        enc5 = self.enc5(enc4)  # H/32 x W/32 x 128

        """ Decoder  """
        up5 = self.up5(enc5)

        add5 = up5
        if self.skip_index[4][0]:
            add5 = add5 + self.skip_down_4_5(self.skip_down_3_4(self.skip_down_2_3(self.skip_down_1_2(data))))
        if self.skip_index[4][1]:
            add5 = add5 + self.skip_down_4_5(self.skip_down_3_4(self.skip_down_2_3(enc1)))
        if self.skip_index[4][2]:
            add5 = add5 + self.skip_down_4_5(self.skip_down_3_4(enc2))
        if self.skip_index[4][3]:
            add5 = add5 + self.skip_down_4_5(enc3)
        if self.skip_index[4][4]:
            add5 = add5 + self.skip5(enc4)

        dec5 = self.dec5(add5)
        up4 = self.up4(dec5)

        add4 = up4
        if self.skip_index[3][0]:
            add4 = add4 + self.skip_down_3_4(self.skip_down_2_3(self.skip_down_1_2(data)))
        if self.skip_index[3][1]:
            add4 = add4 + self.skip_down_3_4(self.skip_down_2_3(enc1))
        if self.skip_index[3][2]:
            add4 = add4 + self.skip_down_3_4(enc2)
        if self.skip_index[3][3]:
            add4 = add4 + self.skip4(enc3)
        if self.skip_index[3][4]:
            add4 = add4 + self.skip_up_5_4(enc4)

        dec4 = self.dec4(add4)
        up3 = self.up3(dec4)

        add3 = up3
        if self.skip_index[2][0]:
            add3 = add3 + self.skip_down_2_3(self.skip_down_1_2(data))
        if self.skip_index[2][1]:
            add3 = add3 + self.skip_down_2_3(enc1)
        if self.skip_index[2][2]:
            add3 = add3 + self.skip3(enc2)
        if self.skip_index[2][3]:
            add3 = add3 + self.skip_up_4_3(enc3)
        if self.skip_index[2][4]:
            add3 = add3 + self.skip_up_4_3(self.skip_up_5_4(enc4))

        dec3 = self.dec3(add3)
        up2 = self.up2(dec3)

        add2 = up2
        if self.skip_index[1][0]:
            add2 = add2 + self.skip_down_1_2(data)
        if self.skip_index[1][1]:
            add2 = add2 + self.skip2(enc1)
        if self.skip_index[1][2]:
            add2 = add2 + self.skip_up_3_2(enc2)
        if self.skip_index[1][3]:
            add2 = add2 + self.skip_up_3_2(self.skip_up_4_3(enc3))
        if self.skip_index[1][4]:
            add2 = add2 + self.skip_up_3_2(self.skip_up_4_3(self.skip_up_5_4(enc4)))

        dec2 = self.dec2(add2)
        up1 = self.up1(dec2)

        add1 = up1
        if self.skip_index[0][0]:
            add1 = add1 + self.skip1(data)
        if self.skip_index[0][1]:
            add1 = add1 + self.skip_up_2_1(enc1)
        if self.skip_index[0][2]:
            add1 = add1 + self.skip_up_2_1(self.skip_up_3_2(enc2))
        if self.skip_index[0][3]:
            add1 = add1 + self.skip_up_2_1(self.skip_up_3_2(self.skip_up_4_3(enc3)))
        if self.skip_index[0][4]:
            add1 = add1 + self.skip_up_2_1(self.skip_up_3_2(self.skip_up_4_3(self.skip_up_5_4(enc4))))

        dec1 = self.dec1(add1)
        out = self.output(dec1)

        return out
