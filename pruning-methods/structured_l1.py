import numpy as np

import torch
import torch.nn as nn

import models

class L1:
    """ L1 based pruning with an optimizer-like interface  """

    # only works with their resnet56

    def __init__(self, model, pruning_rate=0.25, mode="A"):
        """ Init pruning method """
        self.pruning_rate = float(pruning_rate)
        self.model = model
        self.mode = mode

        # init masks
        self.cfg = []
        self.cfg_mask = []

    ################################################
    # Reporting nonzero entries and number of params
    def count_nonzero(self):
        """ Count nonzero elements """
        return sum(mask.sum() for mask in self.masks)

    def numel(self):
        """ Number of elements """
        return int(sum(mask.view(-1).size(0) for mask in self.masks))
    ################################################

    ############################################
    # Methods for resetting or rewinding params
    def clone_params(self):
        """ Copy all tracked params, such that they we can rewind to them later """
        return [p.clone() for p in self.model.parameters()]

    def rewind(self, cloned_params):
        """ Rewind to previously stored params """
        for p_old, p_new in zip(self.model.parameters(), cloned_params):
            p_old.data = p_new.data
    ############################################

    ##############
    # Core methods
    def step(self):
        """ Update the pruning masks """

        skip = {
            'A': [16, 20, 38, 54],
            'B': [16, 18, 20, 34, 38, 54],
        }

        prune_prob = {
            'A': [0.1, 0.1, 0.1],
            'B': [0.6, 0.3, 0.1],
        }

        layer_id = 1
        self.cfg = []
        self.cfg_mask = []
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                out_channels = m.weight.data.shape[0]
                if layer_id in skip[self.mode]:
                    self.cfg_mask.append(torch.ones(out_channels))
                    self.cfg.append(out_channels)
                    layer_id += 1
                    continue
                if layer_id % 2 == 0:
                    if layer_id <= 18:
                        stage = 0
                    elif layer_id <= 36:
                        stage = 1
                    else:
                        stage = 2
                    prune_prob_stage = prune_prob[self.mode][stage]
                    weight_copy = m.weight.data.abs().clone().cpu().numpy()
                    L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                    num_keep = int(out_channels * (1 - prune_prob_stage))
                    arg_max = np.argsort(L1_norm)
                    arg_max_rev = arg_max[::-1][:num_keep]
                    mask = torch.zeros(out_channels)
                    mask[arg_max_rev.tolist()] = 1
                    self.cfg_mask.append(mask)
                    self.cfg.append(num_keep)
                    layer_id += 1
                    continue
                layer_id += 1

    def zero_params(self, masks=None):
        """ Apply the masks, ie, zero out the params """
        masks = masks if masks is not None else self.cfg_mask

        print('zeroing')
        layer_id_in_cfg = 0
        mask = torch.ones(3)
        conv_count = 1

        for m0 in self.model.modules():
            if isinstance(m0, nn.Conv2d):
                print(conv_count)
                if conv_count == 1:
                    conv_count += 1
                    continue
                if conv_count % 2 == 0:
                    mask = masks[layer_id_in_cfg]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))

                    m0.weight.data = m0.weight.data[idx.tolist(), :, :, :]
                    layer_id_in_cfg += 1
                    conv_count += 1
                    continue
                if conv_count % 2 == 1:
                    mask = masks[layer_id_in_cfg - 1]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    m0.weight.data = m0.weight.data[:, idx.tolist(), :, :]
                    conv_count += 1
                    continue
            elif isinstance(m0, nn.BatchNorm2d):
                if conv_count % 2 == 1:
                    mask = masks[layer_id_in_cfg - 1]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    m0.weight.data = m0.weight.data[idx.tolist()]
                    m0.bias.data = m0.bias.data[idx.tolist()]
                    m0.running_mean = m0.running_mean[idx.tolist()]
                    m0.running_var = m0.running_var[idx.tolist()]

    ##############
