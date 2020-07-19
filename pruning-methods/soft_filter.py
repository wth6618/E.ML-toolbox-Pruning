import numpy as np

import torch
import torch.nn as nn

import models


class Soft_filter:
    """ CS pruning with an optimizer-like interface  """

    # as of now, this engine only supports resnet 20 32 56 110


    def __init__(self, model, pruning_rate=0.25, depth=110):
        """ Init pruning method """
        self.pruning_rate = float(pruning_rate)
        self.model = model
        self.depth = depth

        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []

        self.init_length()


    def get_codebook(self, weight_torch,compress_rate,length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int (length * (1-compress_rate) )]
        weight_np [weight_np <= -threshold  ] = 1
        weight_np [weight_np >= threshold  ] = 1
        weight_np [weight_np !=1  ] = 0

        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch,compress_rate,length):
        codebook = np.ones(length)
        if len( weight_torch.size())==4:
            filter_pruned_num = int(weight_torch.size()[0]*(1-compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0],-1)
            norm2 = torch.norm(weight_vec,2,1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            kernel_length = weight_torch.size()[1] *weight_torch.size()[2] *weight_torch.size()[3]
            for x in range(0,len(filter_index)):
                codebook [filter_index[x] *kernel_length : (filter_index[x]+1) *kernel_length] = 0

            print("filter codebook done")
        else:
            pass
        return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, layer_rate):
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
        for key in range(1, self.depth + 1, 1):
            self.compress_rate[key] = layer_rate
        # different setting for  different architecture
        if self.depth == 20:
            last_index = 57
        elif self.depth == 32:
            last_index = 93
        elif self.depth == 56:
            last_index = 165
        elif self.depth == 110:
            last_index = 327
        self.mask_index = [x for x in range(0, last_index, 3)]

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            if (index == 0):
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))
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
        self.init_rate(1 - self.pruning_rate)
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index], self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                self.mat[index] = self.mat[index].cuda()
        print("mask Ready")

    def zero_params(self, masks=None):
        """ Apply the masks, ie, zero out the params """
        for index, item in enumerate(self.model.parameters()):
            if(index in self.mask_index):
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        print("mask Done")

    ##############
