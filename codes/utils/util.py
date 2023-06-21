import os
import sys
import time
import math
import torch.nn.functional as F
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from shutil import get_terminal_size

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


####################
# image convert
####################
def crop_border(img_list, crop_border):
    """Crop borders of images
    Args:
        img_list (list [Numpy]): HWC
        crop_border (int): crop border for each end of height and weight

    Returns:
        (list [Numpy]): cropped image list
    """
    if crop_border == 0:
        return img_list
    else:
        return [v[crop_border:-crop_border, crop_border:-crop_border] for v in img_list]


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        #img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        img_np = np.transpose(img_np[:, :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        img_np = np.transpose(img_np[:, :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    elif out_type == np.uint16:
        img_np = (img_np * 1023.0).round()
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def single_forward(model, inp):
    """PyTorch model forward (single test), it is just a simple warpper
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    with torch.no_grad():
        model_output = model(inp)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    output = output.data.float().cpu()
    return output

def single_forward_gbrwnn(model, inp, nf, scale):
    """PyTorch model forward (single test), it is just a simple warpper
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    with torch.no_grad():
        input_LR = inp.permute(0, 2, 1, 3, 4)

        first = False
        last = False

        l0_idx_list = [0, 1, 1, 2, 2]
        c_idx_list = [1, 5, 2, 4, 3]
        l1_idx_list = [6, 6, 5, 5, 4]

        for l0_idx, c_idx, l1_idx in zip(l0_idx_list, c_idx_list, l1_idx_list):
            if c_idx == 1:
                first = True
                l0_temp = torch.zeros_like(input_LR[:, 0:1, 0, :, :])
                l0_o = l0_temp.repeat(1, scale * scale * 3, 1, 1)
                l0_h = l0_temp.repeat(1, nf, 1, 1)

                l1_temp = torch.zeros_like(input_LR[:, 0:1, 0, :, :])
                l1_o = l1_temp.repeat(1, scale * scale * 3, 1, 1)
                l1_h = l1_temp.repeat(1, nf, 1, 1)

                hidden, prediction = model(input_LR[:, :, c_idx, :, :],
                                           input_LR[:, :, l0_idx, :, :], l0_h, l0_o,
                                           input_LR[:, :, l1_idx, :, :], l1_h, l1_o,
                                           first, last)

                r0_hidden = hidden
                r0_prediction = prediction

                first = False
            elif c_idx == 5:
                last = True
                l1_temp = torch.zeros_like(input_LR[:, 0:1, 0, :, :])
                l1_o = l1_temp.repeat(1, scale * scale * 3, 1, 1)
                l1_h = l1_temp.repeat(1, nf, 1, 1)

                hidden, prediction = model(input_LR[:, :, c_idx, :, :],
                                           input_LR[:, :, l0_idx, :, :], r0_hidden,
                                           r0_prediction,
                                           input_LR[:, :, l1_idx, :, :], l1_h, l1_o,
                                           first, last)

                r1_hidden = hidden
                r1_prediction = prediction

                last = False
            elif c_idx == 2:
                hidden, prediction = model(input_LR[:, :, c_idx, :, :],
                                           input_LR[:, :, l0_idx, :, :], r0_hidden,
                                           r0_prediction,
                                           input_LR[:, :, l1_idx, :, :], r1_hidden,
                                           r1_prediction,
                                           first, last)

                r0_hidden = hidden
                r0_prediction = prediction
            elif c_idx == 4:
                hidden, prediction = model(input_LR[:, :, c_idx, :, :],
                                           input_LR[:, :, l0_idx, :, :], r0_hidden,
                                           r0_prediction,
                                           input_LR[:, :, l1_idx, :, :], r1_hidden,
                                           r1_prediction,
                                           first, last)

                r1_hidden = hidden
                r1_prediction = prediction
            elif c_idx == 3:
                hidden, prediction = model(input_LR[:, :, c_idx, :, :],
                                           input_LR[:, :, l0_idx, :, :], r0_hidden,
                                           r0_prediction,
                                           input_LR[:, :, l1_idx, :, :], r1_hidden,
                                           r1_prediction,
                                           first, last)

                model_output = prediction

        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    output = output.data.float().cpu()
    return output

def flipx4_forward(model, inp):
    """Flip testing with X4 self ensemble, i.e., normal, flip H, flip W, flip H and W
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    # normal
    output_f = single_forward(model, inp)

    # flip W
    output = single_forward(model, torch.flip(inp, (-1, )))
    output_f = output_f + torch.flip(output, (-1, ))
    # flip H
    output = single_forward(model, torch.flip(inp, (-2, )))
    output_f = output_f + torch.flip(output, (-2, ))
    # flip both H and W
    output = single_forward(model, torch.flip(inp, (-2, -1)))
    output_f = output_f + torch.flip(output, (-2, -1))

    return output_f / 4


####################
# metric
####################


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_psnr_10bit(img1, img2):
    # img1 and img2 have range [0, 1023]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    # return 20 * math.log10(1023 / math.sqrt(mse))
    return 10 * math.log10(1023**2 / mse)


def ssim(img1, img2, maxv=255):
    C1 = (0.01 * maxv)**2
    C2 = (0.03 * maxv)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_ssim_10bit(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 1023]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2, maxv=1023)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2, maxv=1023))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2), maxv=1023)
    else:
        raise ValueError('Wrong input image dimensions.')

class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()
