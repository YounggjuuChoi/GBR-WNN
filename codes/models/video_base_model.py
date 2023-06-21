import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.model_definition as network
from torch.optim import lr_scheduler
import models.lr_scheduler as custom_lr_scheduler
from .base_model import BaseModel
from models.loss_function import CharbonnierLoss

logger = logging.getLogger('base')


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.net = network.define_model(opt).to(self.device)
        if opt['dist']:
            self.net = DistributedDataParallel(self.net, device_ids=[torch.cuda.current_device()])
        else:
            self.net = DataParallel(self.net)

        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.net.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))

            self.l_pix_w = train_opt['pixel_weight']

            #### optimizers
            wd = train_opt['weight_decay'] if train_opt['weight_decay'] else 0

            optim_params = []
            for k, v in self.net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer = torch.optim.Adam(optim_params, lr=train_opt['lr'],
                                              weight_decay=wd,
                                              betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=train_opt['scheduler_milestones'],
                                                 gamma=train_opt['scheduler_gamma']))
            elif train_opt['lr_scheme'] == 'MultiStepLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        custom_lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        custom_lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LRs'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        opt_net = self.opt['network']
        scale = opt_net['scale']
        nf = opt_net['nf']

        self.var_L = self.var_L.permute(0, 2, 1, 3, 4)

        first = False
        last = False

        self.optimizer.zero_grad()

        l0_idx_list = [0, 1, 1, 2, 2]
        c_idx_list = [1, 5, 2, 4, 3]
        l1_idx_list = [6, 6, 5, 5, 4]

        for l0_idx, c_idx, l1_idx in zip(l0_idx_list, c_idx_list, l1_idx_list):
            if c_idx == 1:
                first = True
                l0_temp = torch.zeros_like(self.var_L[:, 0:1, 0, :, :])
                l0_o = l0_temp.repeat(1, scale * scale * 3, 1, 1)
                l0_h = l0_temp.repeat(1, nf, 1, 1)

                l1_temp = torch.zeros_like(self.var_L[:, 0:1, 0, :, :])
                l1_o = l1_temp.repeat(1, scale * scale * 3, 1, 1)
                l1_h = l1_temp.repeat(1, nf, 1, 1)

                self.hidden, self.prediction = self.net(self.var_L[:, :, c_idx, :, :],
                                                         self.var_L[:, :, l0_idx, :, :], l0_h, l0_o,
                                                         self.var_L[:, :, l1_idx, :, :], l1_h, l1_o,
                                                         first, last)

                self.r0_hidden = self.hidden
                self.r0_prediction = self.prediction

                first = False
            elif c_idx == 5:
                last = True
                l1_temp = torch.zeros_like(self.var_L[:, 0:1, 0, :, :])
                l1_o = l1_temp.repeat(1, scale * scale * 3, 1, 1)
                l1_h = l1_temp.repeat(1, nf, 1, 1)

                self.hidden, self.prediction = self.net(self.var_L[:, :, c_idx, :, :],
                                                         self.var_L[:, :, l0_idx, :, :], self.r0_hidden,
                                                         self.r0_prediction,
                                                         self.var_L[:, :, l1_idx, :, :], l1_h, l1_o,
                                                         first, last)

                self.r1_hidden = self.hidden
                self.r1_prediction = self.prediction

                last = False
            elif c_idx == 2:
                self.hidden, self.prediction = self.net(self.var_L[:, :, c_idx, :, :],
                                                         self.var_L[:, :, l0_idx, :, :], self.r0_hidden,
                                                         self.r0_prediction,
                                                         self.var_L[:, :, l1_idx, :, :], self.r1_hidden,
                                                         self.r1_prediction,
                                                         first, last)

                self.r0_hidden = self.hidden
                self.r0_prediction = self.prediction
            elif c_idx == 4:
                self.hidden, self.prediction = self.net(self.var_L[:, :, c_idx, :, :],
                                                         self.var_L[:, :, l0_idx, :, :], self.r0_hidden,
                                                         self.r0_prediction,
                                                         self.var_L[:, :, l1_idx, :, :], self.r1_hidden,
                                                         self.r1_prediction,
                                                         first, last)

                self.r1_hidden = self.hidden
                self.r1_prediction = self.prediction
            elif c_idx == 3:
                self.hidden, self.prediction = self.net(self.var_L[:, :, c_idx, :, :],
                                                         self.var_L[:, :, l0_idx, :, :], self.r0_hidden,
                                                         self.r0_prediction,
                                                         self.var_L[:, :, l1_idx, :, :], self.r1_hidden,
                                                         self.r1_prediction,
                                                         first, last)

                self.fake_H = self.prediction

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.net.eval()
        opt_net = self.opt['network']

        with torch.no_grad():
            scale = opt_net['scale']
            nf = opt_net['nf']

            self.var_L = self.var_L.permute(0, 2, 1, 3, 4)

            first = False
            last = False

            l0_idx_list = [0, 1, 1, 2, 2]
            c_idx_list = [1, 5, 2, 4, 3]
            l1_idx_list = [6, 6, 5, 5, 4]

            for l0_idx, c_idx, l1_idx in zip(l0_idx_list, c_idx_list, l1_idx_list):
                if c_idx == 1:
                    first = True
                    l0_temp = torch.zeros_like(self.var_L[:, 0:1, 0, :, :])
                    l0_o = l0_temp.repeat(1, scale * scale * 3, 1, 1)
                    l0_h = l0_temp.repeat(1, nf, 1, 1)

                    l1_temp = torch.zeros_like(self.var_L[:, 0:1, 0, :, :])
                    l1_o = l1_temp.repeat(1, scale * scale * 3, 1, 1)
                    l1_h = l1_temp.repeat(1, nf, 1, 1)

                    self.hidden, self.prediction = self.net(self.var_L[:, :, c_idx, :, :],
                                                             self.var_L[:, :, l0_idx, :, :], l0_h, l0_o,
                                                             self.var_L[:, :, l1_idx, :, :], l1_h, l1_o,
                                                             first, last)

                    self.r0_hidden = self.hidden
                    self.r0_prediction = self.prediction

                    first = False
                elif c_idx == 5:
                    last = True
                    l1_temp = torch.zeros_like(self.var_L[:, 0:1, 0, :, :])
                    l1_o = l1_temp.repeat(1, scale * scale * 3, 1, 1)
                    l1_h = l1_temp.repeat(1, nf, 1, 1)

                    self.hidden, self.prediction = self.net(self.var_L[:, :, c_idx, :, :],
                                                             self.var_L[:, :, l0_idx, :, :], self.r0_hidden,
                                                             self.r0_prediction,
                                                             self.var_L[:, :, l1_idx, :, :], l1_h, l1_o,
                                                             first, last)

                    self.r1_hidden = self.hidden
                    self.r1_prediction = self.prediction

                    last = False
                elif c_idx == 2:
                    self.hidden, self.prediction = self.net(self.var_L[:, :, c_idx, :, :],
                                                             self.var_L[:, :, l0_idx, :, :], self.r0_hidden,
                                                             self.r0_prediction,
                                                             self.var_L[:, :, l1_idx, :, :], self.r1_hidden,
                                                             self.r1_prediction,
                                                             first, last)

                    self.r0_hidden = self.hidden
                    self.r0_prediction = self.prediction
                elif c_idx == 4:
                    self.hidden, self.prediction = self.net(self.var_L[:, :, c_idx, :, :],
                                                             self.var_L[:, :, l0_idx, :, :], self.r0_hidden,
                                                             self.r0_prediction,
                                                             self.var_L[:, :, l1_idx, :, :], self.r1_hidden,
                                                             self.r1_prediction,
                                                             first, last)

                    self.r1_hidden = self.hidden
                    self.r1_prediction = self.prediction
                elif c_idx == 3:
                    self.hidden, self.prediction = self.net(self.var_L[:, :, c_idx, :, :],
                                                             self.var_L[:, :, l0_idx, :, :], self.r0_hidden,
                                                             self.r0_prediction,
                                                             self.var_L[:, :, l1_idx, :, :], self.r1_hidden,
                                                             self.r1_prediction,
                                                             first, last)

                    self.fake_H = self.prediction
        self.net.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.net)
        if isinstance(self.net, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.net.__class__.__name__,
                                             self.net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.net.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path = self.opt['path']['pretrain_model']
        if load_path is not None:
            logger.info('Loading model for [{:s}] ...'.format(load_path))
            self.load_network(load_path, self.net, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.net, iter_label)
