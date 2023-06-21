""" Test for GBR-WNN """

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import datasets.util as data_util
import models.archs.GBRWNN_arch as GBRWNN_arch

def main():
    ### settings
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_mode = 'Vid4' # Vid4 | sharp_bicubic | SPMCS | DAVIS-2019
    model_mode = 'L' # L | M | S
    davis_res = '1280x720'  # 1920x1080 | 1280x720
    flip_test = False
    save_imgs = False
    except_num = 2  # |0|1|2|
    channel_opt = 'y'  # |y|rgb|
    padding = 'new_info'

    ### pretrained model path
    if model_mode == 'L':
        model_path = '../pretrained_models/GBR-WNN-L.pth'
    elif model_mode == 'M':
        model_path = '../pretrained_models/GBR-WNN-M.pth'
    elif model_mode == 'S':
        model_path = '../pretrained_models/GBR-WNN-S.pth'
    else:
        raise NotImplementedError

    ### model arch
    N_in = 7
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    nf = 128
    RBs = 30
    if model_mode == 'L':
        RBs = 30
    elif model_mode == 'M':
        RBs = 20
    elif model_mode == 'S':
        RBs = 10
    scale = 4
    model = GBRWNN_arch.GBRWNN(nf, N_in, RBs, scale)

    ### dataset
    if data_mode == 'Vid4':
        test_dataset_folder = '../datasets/Vid4/BIx4'
        GT_dataset_folder = '../datasets/Vid4/GT'
    elif data_mode == 'SPMCS':
        test_dataset_folder = '../datasets/SPMCS/BIx4'
        GT_dataset_folder = '../datasets/SPMCS/GT_crop'
    elif data_mode == 'DAVIS-2019':
        test_dataset_folder = '../datasets/{}/BIx4/{}'.format(data_mode, davis_res)
        GT_dataset_folder = '../datasets/{}/GT/{}'.format(data_mode, davis_res)
    else:
        raise NotImplementedError

    ### save settings
    save_folder = '../experiments/test_results/results_{}/{}'.format(data_mode, model_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    ### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model: {} - {}'.format(model_mode, model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Channel: {}'.format(channel_opt))
    logger.info('Except {}frames'.format(str(except_num)))

    ### load model
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    model = model.to(device)

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l, avg_ssim_l = [], [], [], []
    subfolder_name_l = []

    subfolder_l = sorted(filter(os.path.isdir, glob.glob(osp.join(test_dataset_folder, '*'))))
    subfolder_GT_l = sorted(filter(os.path.isdir, glob.glob(osp.join(GT_dataset_folder, '*'))))
    # for each subfolder
    for subfolder, subfolder_GT in zip(subfolder_l, subfolder_GT_l):
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        imgs_LQ = data_util.read_img_seq(subfolder)
        img_GT_l = []
        for img_GT_path in sorted(glob.glob(osp.join(subfolder_GT, '*'))):
            img_GT_l.append(data_util.read_img(None, img_GT_path))

        avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center, avg_ssim = 0, 0, 0, 0, 0, 0

        for img_idx, img_path in enumerate(img_path_l):
            if img_idx > (except_num - 1) and img_idx < (max_idx - except_num):
                img_name = osp.splitext(osp.basename(img_path))[0]
                select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
                imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

                output = util.single_forward_gbrwnn(model, imgs_in, nf, scale)
                output = util.tensor2img(output.squeeze(0))

                if save_imgs:
                    cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

                # calculate PSNR
                output = output / 255.
                GT = np.copy(img_GT_l[img_idx])

                if channel_opt == 'y':  # bgr2y, [0, 1]
                    GT = data_util.bgr2ycbcr(GT, only_y=True)
                    output = data_util.bgr2ycbcr(output, only_y=True)

                output, GT = util.crop_border([output, GT], crop_border)
                crt_psnr = util.calculate_psnr(output * 255, GT * 255)
                crt_ssim = util.calculate_ssim(output * 255, GT * 255)
                logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB'.format(img_idx + 1, img_name, crt_psnr))
                logger.info('{:3d} - {:25} \tSSIM: {:.6f}'.format(img_idx + 1, img_name, crt_ssim))

                if img_idx >= border_frame and img_idx < max_idx - border_frame:  # center frames
                    avg_psnr_center += crt_psnr
                    N_center += 1
                else:  # border frames
                    avg_psnr_border += crt_psnr
                    N_border += 1

                avg_ssim += crt_ssim

        avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        avg_psnr_center = avg_psnr_center / N_center
        avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_psnr_border_l.append(avg_psnr_border)

        avg_ssim = avg_ssim / (N_center + N_border)
        avg_ssim_l.append(avg_ssim)

        logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
                    'Center PSNR: {:.6f} dB for {} frames; '
                    'Border PSNR: {:.6f} dB for {} frames.'.format(subfolder_name, avg_psnr,
                                                                   (N_center + N_border),
                                                                   avg_psnr_center, N_center,
                                                                   avg_psnr_border, N_border))

        logger.info('Folder {} - Average SSIM: {:.6f} dB for {} frames; '.format(subfolder_name, avg_ssim,
                                                                                 (N_center + N_border)))

    logger.info('################ Tidy Outputs ################')
    for subfolder_name, psnr, psnr_center, psnr_border, ssim in zip(subfolder_name_l, avg_psnr_l,
                                                                    avg_psnr_center_l, avg_psnr_border_l, avg_ssim_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB. '
                    'Center PSNR: {:.6f} dB. '
                    'Border PSNR: {:.6f} dB.'.format(subfolder_name, psnr, psnr_center,
                                                     psnr_border))
        logger.info('Folder {} - Average SSIM: {:.6f}.'.format(subfolder_name, ssim))

    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
                'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB.'.format(
        sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_l),
        sum(avg_psnr_center_l) / len(avg_psnr_center_l),
        sum(avg_psnr_border_l) / len(avg_psnr_border_l)))
    logger.info('Total Average SSIM: {:.6f} for {} clips.'.format(
        sum(avg_ssim_l) / len(avg_ssim_l), len(subfolder_l)))

if __name__ == '__main__':
    main()