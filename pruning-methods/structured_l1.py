import numpy as np

import torch
import torch.nn as nn


class L1():
    """ Magnitude pruning with an optimizer-like interface  """

    def __init__(self, model, pruning_rate=0.25):
        """ Init pruning method """
        self.pruning_rate = float(pruning_rate)
        self.model = model

        # init masks
        self.masks = []
        self.cfg = []

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

        self.cfg = []
        self.masks = []
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                out_channels = m.weight.data.shape[0]
                prune_prob_stage = self.pruning_rate
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                num_keep = int(out_channels * (1 - prune_prob_stage))
                arg_max = np.argsort(L1_norm)
                arg_max_rev = arg_max[::-1][:num_keep]
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                self.masks.append(mask)
                self.cfg.append(num_keep)

    def zero_params(self, masks=None):
        """ Apply the masks, ie, zero out the params """
        layer_id_in_cfg = 0
        conv_count = 1
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                if conv_count == 1:
                    conv_count += 1
                    continue
                if conv_count % 2 == 0:
                    mask = self.masks[layer_id_in_cfg]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    layer.weight.data = layer.weight.data[idx.tolist(), :, :, :]
                    layer_id_in_cfg += 1
                    conv_count += 1
                    continue
                if conv_count % 2 == 1:
                    mask = self.masks[layer_id_in_cfg - 1]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    layer.weight.data = layer.weight.data[:, idx.tolist(), :, :]
                    conv_count += 1
                    continue
            elif isinstance(layer, nn.BatchNorm2d):
                if conv_count % 2 == 1:
                    mask = self.masks[layer_id_in_cfg - 1]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    layer.weight.data = layer.weight.data[idx.tolist()]
                    layer.bias.data = layer.bias.data[idx.tolist()]
                    layer.running_mean = layer.running_mean[idx.tolist()]
                    layer.running_var = layer.running_var[idx.tolist()]
                    continue

    ##############
