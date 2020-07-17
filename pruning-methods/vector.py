import numpy as np

import torch
import torch.nn as nn


class Vector:
    """ Vector pruning with an optimizer-like interface  """

    def __init__(self, model, pruning_rate=0.25):
        """ Init pruning method """
        self.pruning_rate = float(pruning_rate)
        self.model = model

        # init masks
        self.masks = []

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

        self.masks = []
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                total = layer.out_channels * layer.kernel_size[0]
                row_sum = layer.weight.data.abs().clone().sum((1, 3), keepdim=True)
                flat, indice = torch.sort(row_sum.flatten())
                threshold = flat[int(np.ceil(int(total) * self.pruning_rate))]

                m_weight = layer.weight.data.abs().clone()
                mask = m_weight.sum((1, 3), keepdim=True).gt(threshold).float().cuda()
                self.masks.append(mask)

    def zero_params(self, masks=None):
        """ Apply the masks, ie, zero out the params """
        sks = masks if masks is not None else self.masks
        conv_count = 0
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                layer.weight.data.mul_(masks[conv_count])
                conv_count += 1

    ##############
