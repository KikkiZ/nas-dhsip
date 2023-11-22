import argparse
import os

import numpy as np
import scipy.io as sio
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.common_utils import print_images, get_noise, get_params, optimize

data_type = torch.cuda.FloatTensor


def parse_args():
    parser = argparse.ArgumentParser(description='nas-dhsip')

    parser.add_argument('--optimizer', dest='optimizer', default='adam', type=str)
    parser.add_argument('--num_iter', dest='num_iter', default=3000, type=int)  # the number of iterations
    parser.add_argument('--show_every', dest='show_every', default=50, type=int)
    parser.add_argument('--lr', dest='lr', default=0.01, type=float)
    parser.add_argument('--plot', dest='plot', default=False, type=bool)
    parser.add_argument('--noise_method', dest='noise_method', default='noise', type=str)
    parser.add_argument('--input_depth', dest='input_depth', default=32, type=int)
    parser.add_argument('--output_path', dest='output_path', default='results/denoising', type=str)
    parser.add_argument('--batch_size', dest='batch_size', default=1, type=int)
    parser.add_argument('--random_seed', dest='random_seed', default=0, type=int)
    parser.add_argument('--net', dest='net', default='default', type=str)
    parser.add_argument('--reg_noise_std', dest='reg_noise_std', default=1. / 30., type=float)
    parser.add_argument('--sigma', dest='sigma', default=25, type=float)
    parser.add_argument('--i_nas', dest='i_nas', default=-1, type=int)
    parser.add_argument('--save_png', dest='save_png', default=0, type=int)
    parser.add_argument('--exp_weight', dest='exp_weight', default=0.99, type=float)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.net == 'default':
        global_path = args.output_path + '_' + args.net
    elif args.net == 'nas':
        global_path = args.output_path + '_' + args.net + '_' + str(args.i_nas)
    else:
        assert False, 'Please choose between default and nas'

    # create the output_path if not exists
    if not os.path.exists(global_path):
        os.makedirs(global_path)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    file_name = 'data/denoising.mat'
    mat = sio.loadmat(file_name)
    image = mat['image']
    decrease_image = mat['image_noisy']

    image = torch.from_numpy(image).type(data_type)
    decrease_image = torch.from_numpy(decrease_image).type(data_type)
    print_images(image, decrease_image)

    copy_image = image.clone().detach()
    copy_decrease_image = decrease_image.clone().detach()

    if args.net == 'default':
        from models.skip import skip

        # build the network
        net = skip(image.shape[0],
                   image.shape[0],
                   num_channels_up=[128] * 5,
                   num_channels_down=[128] * 5,
                   num_channels_skip=[4] * 5,
                   filter_size_up=3,
                   filter_size_down=3,
                   filter_size_skip=1,
                   upsample_mode='bilinear',
                   need1x1_up=False,
                   need_sigmoid=False,
                   need_bias=True,
                   pad='reflection',
                   act_fun='LeakyReLU')
    elif args.net == 'nas':
        from models.skip_search import skip

        net = skip(model_index=args.i_NAS,
                   input_channels=args.input_depth,
                   output_channels=args.input_depth,
                   num_channels_down=[128] * 5,
                   num_channels_up=[128] * 5,
                   num_channels_skip=[4] * 5,
                   downsample_mode='stride',
                   need_sigmoid=True,
                   need_bias=True,
                   pad='reflection',
                   act_fun='LeakyReLU')
    else:
        assert False

    net = net.type(data_type)

    # extend the dimension of the tensor
    # from [191, 200, 200] to [1, 191, 200, 200]
    decrease_image = decrease_image[None, :].cuda()
    # generates a noise tensor of a specified size
    net_input = get_noise(image.shape[0], '2D', (image.shape[1], image.shape[2])).type(data_type).detach()
    net_input_saved = net_input.detach().clone()  # clone the noise tensor without grad
    noise = net_input.detach().clone()  # clone twice

    # loss function
    loss_func = torch.nn.MSELoss().type(data_type)
    i = 0                                       # the number of iterations of the model
    out_avg = None                              # the output from the previous iteration
    last_net = None                             # the parameters during model iteration
    psnr_noisy_last = 0                         # psnr of the previous iteration
    exp_weight = args.exp_weight
    show_every = args.show_every
    reg_noise_std = args.reg_noise_std
    writer = SummaryWriter('./logs/denoising')  # the location where the data record is saved


    def closure():
        # declare the following variables as global variables
        global i, out_avg, psnr_noisy_last, last_net, net_input

        # normal_() is used to generate a tensor of a specified size that follows a normal distribution
        # this step adds some random noise to the noise tensor, preventing the model from overfitting
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)  # the result of a network iteration

        # smoothing
        # combine the results of the previous run with this one
        # which can make the results more stable
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        total_loss = loss_func(out, decrease_image)  # calculate the loss value of the loss function
        total_loss.backward()  # back propagation gradient calculation

        psnr_noisy = psnr_gpu(copy_decrease_image, out.squeeze())
        psnr_gt = psnr_gpu(copy_image, out.squeeze())
        psnr_gt_sm = psnr_gpu(copy_image, out_avg.squeeze())

        writer.add_scalar('compare with de', psnr_noisy, i)
        writer.add_scalar('compare with gt', psnr_gt, i)
        writer.add_scalar('compare with gt_sm', psnr_gt_sm, i)

        # backtracking
        if i % show_every == 0:
            out = torch.clamp(out, 0, 1)
            out_avg = torch.clamp(out_avg, 0, 1)

            out_normalize = max_min_normalize(out.squeeze().detach())
            out_avg_normalize = max_min_normalize(out_avg.squeeze().detach())
            print_images(out_normalize, out_avg_normalize)

            if psnr_noisy - psnr_noisy_last < -5:  # model produced an overfit
                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.detach().copy_(new_param.cuda())  # copy the paras from saved to model

                return total_loss * 0  # clean the loss
            else:  # psnr volatility is still within expectations
                last_net = [x.detach().cpu() for x in net.parameters()]  # save the parameters in the model
                psnr_noisy_last = psnr_noisy  # renew psnr_noisy_last

        i += 1

        return total_loss


    params = get_params('net', net, net_input)
    optimize('adam', params, closure, args.lr, args.num_iter)
    writer.close()

    print('finish optimization')
