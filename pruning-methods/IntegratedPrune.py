from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models
from utils import *

cuda = True
def test(model, dataset = 'cifar10'):
	model.eval()
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			if cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)
			output = model(data)
			pred = output.data.max(1, keepdim = True)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
		correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
	return correct / float(len(test_loader.dataset))

def CSPrune(save_path, test_loader,model, percentile, Net = 'vgg'):
    test(model)

    if not os.path.exists(save_path):
        print("path does not exist")
        os.makedirs(save_path)
        
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * percentile)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)) if int(torch.sum(mask)) != 0 else 1)
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned/total

    print('Pre-processing Successful!')

    # simple test model after Pre-processing prune (simple set BN scales to zeros)

    acc = test(model)
    #exit()
    # Make real prune
    print(cfg)
    newmodel = models.__dict__[Net](dataset=DATASET, cfg=cfg)
    if cuda:
        newmodel.cuda()

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    savepath = os.path.join(save_path, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n"+str(cfg)+"\n")
        fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
        fp.write("Test accuracy: \n"+str(acc))
    old_modules = list(model.modules())
    new_modules = list(newmodel.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if isinstance(old_modules[layer_id + 1], models.channel_selection):
                # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if isinstance(old_modules[layer_id-1], models.channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                # If the current convolution is not the last convolution in the residual block, then we can change the 
                # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                if conv_count % 3 != 1:
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

            # We need to consider the case where there are downsampling convolutions. 
            # For these convolutions, we just copy the weights.
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(save_path, 'pruned.pth.tar'))

    #print(newmodel)
    model = newmodel
    test(model)
    return model

def UnstructuredPruning(model, percentile, save_path):
    print("Unstructured Pruning")
    if not os.path.exists(save_path):
        print("path does not exist")
        os.makedirs(save_path)
    
    
    #print("------------before pruning-------------")
    #print(layer.weight.data)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            total = layer.weight.data.numel()
            weight_matrix = layer.weight.data.view(-1).abs().clone()
            
            sorted_weight, _ = torch.sort(weight_matrix)
            # print(sorted_weight)
            # print("total")
            # exit()
            threshold = sorted_weight[int(np.ceil(total * percentile))]
            #print(f'total: {total}, threshold: {threshold} ')
            m_weight = layer.weight.data.abs().clone()
            mask = m_weight.gt(threshold).float().cuda()
            #print(mask)
            layer.weight.data.mul_(mask)

    acc = test(model)
    
    path = os.path.join(save_path, "accuracy.txt")
    with open(path, "w+") as temp:
        temp.write("Test accuracy: \n"+str(acc))
    torch.save({'state_dict': model.state_dict()}, os.path.join(save_path, 'unstructured.pth.tar'))
    #print("------------after pruning-------------")
    return acc
    
def VectorPruning(model, percentile, save_path):
    print("Vector Pruning")

    if not os.path.exists(save_path):
        print("path does not exist")
        os.makedirs(save_path)

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            total = layer.out_channels * layer.kernel_size[0]
            #print("total is :",total)
            row_sum = layer.weight.data.abs().clone().sum((1,3), keepdim = True)
            #print(row_sum)
            flat, indice = torch.sort(row_sum.flatten())
            threshold = flat[int(np.ceil(int(total) * percentile))]
            
            m_weight = layer.weight.data.abs().clone()
            mask = m_weight.sum((1,3), keepdim = True).gt(threshold).float().cuda()
            #print(mask)
            layer.weight.data.mul_(mask)

    acc = test(model)
    path = os.path.join(save_path, "accuracy.txt")
    with open(path,"w+") as temp:
        temp.write("Test accuracy: \n"+str(acc))
    torch.save({'state_dict': model.state_dict()}, os.path.join(save_path, 'unstructured.pth.tar'))
    return acc


train_loader, test_loader = DataPreprocess()

#PreTrain('./logs_resNet_refine0.6',_PRUNEDPATH = './PrunedResults_resNet_0.6/pruned.pth.tar', _MODELPATH ='./logs_resNet_refine0.6/checkpoint.pth.tar' ,refine = True, resume = True, Net = 'resnet', epochs = 20 ,depth = 164)
#PreTrain('./logs_resNet_refine0.3',_PRUNEDPATH = './PrunedResults_resNet_0.3/pruned.pth.tar', refine = True,  Net = 'resnet', epochs = 20 ,depth = 164)

#PrintPrunedModel('./PrunedResults_resNet_0.8/pruned.pth.tar', Net = 'resnet', depth = 164)
model = LoadModel('./logs_resNet_refine/checkpoint.pth.tar',_PRUNEDPATH = './PruneResults_resNet/pruned.pth.tar',refine = True,Net = 'resnet', depth = 164)
# print("pruned model")
#model = LoadModel('./PrunedResults_resNet_0.7/pruned.pth.tar',_PRUNEDPATH ='./PrunedResults_resNet_0.7/pruned.pth.tar' ,Net = 'resnet', depth = 164, refine = True)


#model = LoadModel("./logs_resNet/checkpoint.pth.tar",Net = 'resnet', depth = 164)
# test(model)
# UnstructuredPruning(model, 0.2,'./Unstructured_resNet0.2')
# test(model)

# p_dict = [0.2, 0.3,0.4,0.5,0.6,0.7]
# acc1, acc2 =[] , []

# for p in p_dict:
#     print("prune with percentile {}".format(p))

#     path = "./Unstructured_resNet{}".format(str(p))


#     acc1.append(UnstructuredPruning(model, p,path))

# model = LoadModel("./logs_resNet/checkpoint.pth.tar",Net = 'resnet', depth = 164)
# for p in p_dict:
#     print("prune with percentile {}".format(p))
#     path = "./VectorPruning_resNet{}".format(str(p))

#     #print(path)
#     acc2.append(VectorPruning(model, p,path))


# np.savetxt("Accuracy_UV.txt",(np.array(acc1), np.array(acc2)))

#prunedModel = CSPrune('./PrunedResults_resNet_0.2',test_loader,model, 0.2, Net = 'resnet')
# for percentile in np.arange(0.3,0.8, 0.1):
#     print(percentile)

