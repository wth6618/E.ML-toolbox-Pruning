# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 06:08:40 2020

@author: 24100
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(torch.cuda.get_device_name(device))

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 6, 3)
        self.conv3 = nn.Conv2d(6, 8, 3)
        self.fc1 = nn.Linear(8 * 5 * 5, 80)  # 5x5 image dimension
        self.fc2 = nn.Linear(80, 10)
       

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def AllPruningMethod(module,method_list, percentile):
    print(method_list)
    for rank, activation in enumerate(method_list):
        if (rank == 0 and activation):
            FilterPruning(module, percentile)
        elif (rank == 1 and activation):
            VectorPruning(module, percentile)
        elif (rank == 2 and activation):    
            UnstructuredPruning(module, percentile)
        else:
            print("not implemented yet")
            pass
    
    
def UnstructuredPruning(module, percentile):
    print("Unstructured Pruning")
    total = 0
    print("------------before pruning-------------")
    print(module.weight.data)
    total += module.weight.data.numel()
    weight_matrix = module.weight.data.view(-1).abs().clone()
    sorted_weight, _ = torch.sort(weight_matrix)
    #print("sorted:",sorted_weight)
    threshold = sorted_weight[int(np.ceil(total * percentile))]
    print(f'total: {total}, threshold: {threshold} ')
    m_weight = module.weight.data.abs().clone()
    mask = m_weight.gt(threshold).float().cuda()
    #print(mask)
    module.weight.data.mul_(mask)
    print("------------after pruning-------------")
    print(module.weight.data)
    

def FilterPruning(module, percentile):
    
    print("Filter Pruning")
    num_chls = module.out_channels
    print("------------before pruning-------------")
    print(module.weight.data)

    filter_sum = module.weight.data.abs().clone().sum((2,3), keepdim = True)
    #print(filter_sum)

    flat, indice = torch.sort(filter_sum.flatten())
    threshold = flat[int(np.ceil(num_chls * percentile))]
    
    mask_index = np.array([indice[x] for x in range(0,len(flat)) if flat[x] < threshold])
    print(f"threshold:{threshold}")
    print('mask index:',mask_index)    

    
    
    m_weight = module.weight.data.abs().clone()
    mask = m_weight.sum((2,3), keepdim = True).gt(threshold).float().cuda()
    #print(mask)
    module.weight.data.mul_(mask)
    print("------------after pruning-------------")
    print(module.weight.data)

    
    
def VectorPruning(module, percentile):
    print("Vector Pruning")
    print("------------before pruning-------------")
    print(module.weight.data)
    total = module.out_channels * module.kernel_size[0]
    #print("total is :",total)
    row_sum = module.weight.data.abs().clone().sum((1,3), keepdim = True)
    #print(row_sum)
    flat, indice = torch.sort(row_sum.flatten())
    threshold = flat[int(np.ceil(int(total) * percentile))]
    
    m_weight = module.weight.data.abs().clone()
    mask = m_weight.sum((1,3), keepdim = True).gt(threshold).float().cuda()
    #print(mask)
    module.weight.data.mul_(mask)
    print("------------after pruning-------------")
    print(module.weight.data)
    





#print(model.fc1.weight.shape)
#print(model.conv1.kernel_size)


#prune.random_unstructured(module, name="weight", amount=0.3)
#print("-----------after pruning-------------")
#print(list(module.named_buffers()))
#print(module.weight)
#print(list(module.named_parameters()))

def simple_pruning(model:nn.modules, method_list:np.array, percentile: float, parameter = 'weight'):
    #module = model.conv1
    
    #print(list(module.named_parameters()))
    #print(model.conv1.weight.shape)
 
    for name, method in method_list:
        print("pruning layer:",name)
        for layer, module in model.named_modules():
            if name == layer:
                #print("found it, method:", method)
                AllPruningMethod(module, method, percentile)
                break


    
    
'''TEST SECTION '''
model = LeNet().to(device=device)
#for layer, module in model.named_modules():
#    if layer == 'conv1' or layer == 'conv2':
#        print(f'----------{layer}----------')
#        print(module.weight.data)
method_list = np.array([("conv1",[1,1,1]),("conv2",[1,0,0]),("conv3",[1,0,1])])
#method_list = np.array([("conv1",[1,1,1])])

simple_pruning(model,method_list, 0.10)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    