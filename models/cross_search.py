import torch
from torch import nn

from models import layer_generator as generator
from utils import gene_utils


class UNet(nn.Module):

    def __init__(self,
                 input_channel=3,
                 output_channel=3,
                 num_channels_down=None,
                 num_channels_up=None,
                 num_channels_skip=4,
                 genotype=None,
                 kernel_size_down=3,
                 kernel_size_up=3,
                 kernel_size_skip=3,
                 need_sigmoid=True,
                 downsample_mode='stride',
                 activate='LeakyReLU',
                 need1x1_up=True,
                 need_bias=True,
                 pad='zero'):

        super(UNet, self).__init__()

        if num_channels_up is None:
            num_channels_up = [16, 32, 64, 128, 128]
        if num_channels_down is None:
            num_channels_down = [16, 32, 64, 128, 128]

        if genotype is not None:
            self.genotype = genotype
        else:
            assert False

        # encoder layers
        self.encoder_1 = generator.encoder_layer(input_channel=input_channel,
                                                 output_channel=num_channels_down[0],
                                                 downsample_mode=downsample_mode,
                                                 kernel_size=kernel_size_down,
                                                 activate=activate,
                                                 bias=need_bias,
                                                 pad=pad)

        self.encoder_2 = generator.encoder_layer(input_channel=num_channels_down[0],
                                                 output_channel=num_channels_down[1],
                                                 downsample_mode=downsample_mode,
                                                 kernel_size=kernel_size_down,
                                                 activate=activate,
                                                 bias=need_bias,
                                                 pad=pad)

        self.encoder_3 = generator.encoder_layer(input_channel=num_channels_down[1],
                                                 output_channel=num_channels_down[2],
                                                 downsample_mode=downsample_mode,
                                                 kernel_size=kernel_size_down,
                                                 activate=activate,
                                                 bias=need_bias,
                                                 pad=pad)

        self.encoder_4 = generator.encoder_layer(input_channel=num_channels_down[2],
                                                 output_channel=num_channels_down[3],
                                                 downsample_mode=downsample_mode,
                                                 kernel_size=kernel_size_down,
                                                 activate=activate,
                                                 bias=need_bias,
                                                 pad=pad)

        self.encoder_5 = generator.encoder_layer(input_channel=num_channels_down[3],
                                                 output_channel=num_channels_down[4],
                                                 downsample_mode=downsample_mode,
                                                 kernel_size=kernel_size_down,
                                                 activate=activate,
                                                 bias=need_bias,
                                                 pad=pad)

        # same-scale skip connect layers
        self.skip_1 = generator.skip_layer(input_channel=num_channels_down[0],
                                           output_channel=num_channels_skip,
                                           kernel_size=kernel_size_skip,
                                           activate=activate,
                                           bias=need_bias,
                                           pad=pad)

        self.skip_2 = generator.skip_layer(input_channel=num_channels_down[1],
                                           output_channel=num_channels_skip,
                                           kernel_size=kernel_size_skip,
                                           activate=activate,
                                           bias=need_bias,
                                           pad=pad)

        self.skip_3 = generator.skip_layer(input_channel=num_channels_down[2],
                                           output_channel=num_channels_skip,
                                           kernel_size=kernel_size_skip,
                                           activate=activate,
                                           bias=need_bias,
                                           pad=pad)

        self.skip_4 = generator.skip_layer(input_channel=num_channels_down[3],
                                           output_channel=num_channels_skip,
                                           kernel_size=kernel_size_skip,
                                           activate=activate,
                                           bias=need_bias,
                                           pad=pad)

        self.skip_5 = generator.skip_layer(input_channel=num_channels_down[4],
                                           output_channel=num_channels_skip,
                                           kernel_size=kernel_size_skip,
                                           activate=activate,
                                           bias=need_bias,
                                           pad=pad)

        # downsample layers - cross-scale downsample skip connect
        self.skip_down_2 = generator.downsample_layer(input_channel=num_channels_down[0],
                                                      output_channel=num_channels_down[0],
                                                      kernel_size=kernel_size_skip,
                                                      downsample_mode=downsample_mode,
                                                      activate=activate,
                                                      bias=need_bias,
                                                      pad=pad)

        self.skip_down_3 = generator.downsample_layer(input_channel=num_channels_down[1],
                                                      output_channel=num_channels_down[1],
                                                      kernel_size=kernel_size_skip,
                                                      downsample_mode=downsample_mode,
                                                      activate=activate,
                                                      bias=need_bias,
                                                      pad=pad)

        self.skip_down_4 = generator.downsample_layer(input_channel=num_channels_down[2],
                                                      output_channel=num_channels_down[2],
                                                      kernel_size=kernel_size_skip,
                                                      downsample_mode=downsample_mode,
                                                      activate=activate,
                                                      bias=need_bias,
                                                      pad=pad)

        self.skip_down_2_3 = generator.downsample_layer(input_channel=num_channels_down[0],
                                                        output_channel=num_channels_skip,
                                                        kernel_size=kernel_size_skip,
                                                        downsample_mode=downsample_mode,
                                                        activate=activate,
                                                        bias=need_bias,
                                                        pad=pad)

        self.skip_down_3_4 = generator.downsample_layer(input_channel=num_channels_down[1],
                                                        output_channel=num_channels_skip,
                                                        kernel_size=kernel_size_skip,
                                                        downsample_mode=downsample_mode,
                                                        activate=activate,
                                                        bias=need_bias,
                                                        pad=pad)

        self.skip_down_4_5 = generator.downsample_layer(input_channel=num_channels_down[2],
                                                        output_channel=num_channels_skip,
                                                        kernel_size=kernel_size_skip,
                                                        downsample_mode=downsample_mode,
                                                        activate=activate,
                                                        bias=need_bias,
                                                        pad=pad)

        # upsample layers - cross-scale upsample skip connect
        upsample_gene = gene_utils.get_upsample_gene(genotype)
        self.skip_up_4 = generator.upsample_layer(input_channel=num_channels_down[4],
                                                  output_channel=num_channels_skip,
                                                  upsample_params=upsample_gene)
        self.skip_up_3 = generator.upsample_layer(input_channel=num_channels_down[3],
                                                  output_channel=num_channels_skip,
                                                  upsample_params=upsample_gene)
        self.skip_up_2 = generator.upsample_layer(input_channel=num_channels_down[2],
                                                  output_channel=num_channels_skip,
                                                  upsample_params=upsample_gene)
        self.skip_up_1 = generator.upsample_layer(input_channel=num_channels_down[1],
                                                  output_channel=num_channels_skip,
                                                  upsample_params=upsample_gene)

        # upsample layers: 在解码器前执行
        self.skip_gene = gene_utils.get_skip_gene(genotype)
        x = len([i for i in range(0, 5) if self.skip_gene[i][4] == gene_utils.skip_gene[i][4] == 1])
        self.upsample_5 = generator.upsample_layer(input_channel=num_channels_down[4] + x * num_channels_skip,
                                                   output_channel=num_channels_up[4],
                                                   upsample_params=upsample_gene)
        # print(self.upsample_5)
        x = len([i for i in range(0, 5) if self.skip_gene[i][3] == gene_utils.skip_gene[i][3] == 1])
        self.upsample_4 = generator.upsample_layer(input_channel=num_channels_down[4] + x * num_channels_skip,
                                                   output_channel=num_channels_up[3],
                                                   upsample_params=upsample_gene)
        # print(self.upsample_4)
        x = len([i for i in range(0, 5) if self.skip_gene[i][2] == gene_utils.skip_gene[i][2] == 1])
        self.upsample_3 = generator.upsample_layer(input_channel=num_channels_down[3] + x * num_channels_skip,
                                                   output_channel=num_channels_up[2],
                                                   upsample_params=upsample_gene)
        x = len([i for i in range(0, 5) if self.skip_gene[i][1] == gene_utils.skip_gene[i][1] == 1])
        self.upsample_2 = generator.upsample_layer(input_channel=num_channels_down[2] + x * num_channels_skip,
                                                   output_channel=num_channels_up[1],
                                                   upsample_params=upsample_gene)
        x = len([i for i in range(0, 5) if self.skip_gene[i][0] == gene_utils.skip_gene[i][0] == 1])
        self.upsample_1 = generator.upsample_layer(input_channel=num_channels_down[1] + x * num_channels_skip,
                                                   output_channel=num_channels_up[0],
                                                   upsample_params=upsample_gene)

        # decoder layers
        self.decoder_5 = generator.decoder_layer(input_channel=num_channels_up[4],
                                                 output_channel=num_channels_up[4],
                                                 kernel_size=kernel_size_up,
                                                 need1x1_up=need1x1_up,
                                                 activate=activate,
                                                 bias=need_bias,
                                                 pad=pad)

        self.decoder_4 = generator.decoder_layer(input_channel=num_channels_up[4],
                                                 output_channel=num_channels_up[3],
                                                 kernel_size=kernel_size_up,
                                                 need1x1_up=need1x1_up,
                                                 activate=activate,
                                                 bias=need_bias,
                                                 pad=pad)

        self.decoder_3 = generator.decoder_layer(input_channel=num_channels_up[3],
                                                 output_channel=num_channels_up[2],
                                                 kernel_size=kernel_size_up,
                                                 need1x1_up=need1x1_up,
                                                 activate=activate,
                                                 bias=need_bias,
                                                 pad=pad)

        self.decoder_2 = generator.decoder_layer(input_channel=num_channels_up[2],
                                                 output_channel=num_channels_up[1],
                                                 kernel_size=kernel_size_up,
                                                 need1x1_up=need1x1_up,
                                                 activate=activate,
                                                 bias=need_bias,
                                                 pad=pad)

        self.decoder_1 = generator.decoder_layer(input_channel=num_channels_up[1],
                                                 output_channel=num_channels_up[0],
                                                 kernel_size=kernel_size_up,
                                                 need1x1_up=need1x1_up,
                                                 activate=activate,
                                                 bias=need_bias,
                                                 pad=pad)

        self.output_layer = generator.output_layer(input_channel=num_channels_up[0],
                                                   output_channel=output_channel,
                                                   need_sigmoid=need_sigmoid,
                                                   bias=need_bias,
                                                   pad=pad)

    def forward(self, data):
        # 编码
        enc_1 = self.encoder_1(data)
        enc_2 = self.encoder_2(enc_1)
        enc_3 = self.encoder_3(enc_2)
        enc_4 = self.encoder_4(enc_3)
        enc_5 = self.encoder_5(enc_4)

        # 跳跃连接和解码
        dec_5 = enc_5
        if self.skip_gene[4][4] == 1:
            # add_5 = add_5 + self.skip_5(enc_5)
            dec_5 = torch.cat([dec_5, self.skip_5(enc_5)], dim=1)
        if self.skip_gene[2][4] == 1:
            # add_5 = self.skip_down_4_5(self.skip_down_4(enc_3))
            dec_5 = torch.cat([dec_5, self.skip_down_4_5(self.skip_down_4(enc_3))], dim=1)
        up_5 = self.upsample_5(dec_5)
        dec_4 = self.decoder_5(up_5)

        if self.skip_gene[1][3] == 1:
            # add_4 = self.skip_down_3_4(self.skip_down_3(enc_2))
            dec_4 = torch.cat([dec_4, self.skip_down_3_4(self.skip_down_3(enc_2))], dim=1)
        if self.skip_gene[3][3] == 1:
            # add_4 = add_4 + self.skip_4(enc_4)
            dec_4 = torch.cat([dec_4, self.skip_4(enc_4)], dim=1)
        if self.skip_gene[4][3] == 1:
            # add_4 = add_4 + self.skip_up_4(enc_5)
            dec_4 = torch.cat([dec_4, self.skip_up_4(enc_5)], dim=1)
        up_4 = self.upsample_4(dec_4)
        dec_3 = self.decoder_4(up_4)

        if self.skip_gene[0][2] == 1:
            # add_3 = self.skip_down_2_3(self.skip_down_2(enc_1))
            dec_3 = torch.cat([dec_3, self.skip_down_2_3(self.skip_down_2(enc_1))], dim=1)
        if self.skip_gene[2][2] == 1:
            # add_3 = add_3 + self.skip_3(enc_3)
            dec_3 = torch.cat([dec_3, self.skip_3(enc_3)], dim=1)
        if self.skip_gene[3][2] == 1:
            # add_3 = add_3 + self.skip_up_3(enc_4)
            dec_3 = torch.cat([dec_3, self.skip_up_3(enc_4)], dim=1)
        up_3 = self.upsample_3(dec_3)
        dec_2 = self.decoder_3(up_3)

        if self.skip_gene[1][1] == 1:
            # add_2 = self.skip_2(enc_2)
            dec_2 = torch.cat([dec_2, self.skip_2(enc_2)], dim=1)
        if self.skip_gene[2][1] == 1:
            # add_2 = add_2 + self.skip_up_2(enc_3)
            dec_2 = torch.cat([dec_2, self.skip_up_2(enc_3)], dim=1)
        up_2 = self.upsample_2(dec_2)
        dec_1 = self.decoder_2(up_2)

        if self.skip_gene[0][0] == 1:
            # add_1 = self.skip_1(enc_1)
            dec_1 = torch.cat([dec_1, self.skip_1(enc_1)], dim=1)
        if self.skip_gene[1][0] == 1:
            # add_1 = add_1 + self.skip_up_1(enc_2)
            dec_1 = torch.cat([dec_1, self.skip_up_1(enc_2)], dim=1)
        up_1 = self.upsample_1(dec_1)
        out = self.decoder_1(up_1)

        return self.output_layer(out)
