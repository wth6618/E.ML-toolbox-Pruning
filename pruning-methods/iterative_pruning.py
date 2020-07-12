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
import utils


# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
cuda = torch.cuda.is_available()
print("Cuda Status:", cuda)

def load_model(_modelpath, _PRUNEDPATH='', net='vgg', usegpu=True, depth=19, _dataset='cifar10', refine=False):
    # if not os.path.exists(PRUNEDPATH):
    #     print("path does not exist")
    #     os.makedirs(PRUNEDPATH)

    if refine:
        print("Refining")
        checkpoint = torch.load(_PRUNEDPATH)
        model = models.__dict__[net](dataset=_dataset, depth=depth, cfg=checkpoint['cfg'])
        # model.load_state_dict(checkpoint['state_dict'])
    else:
        model = models.__dict__[net](dataset=_dataset, depth=depth)

    if cuda and usegpu:
        print("LoadModel use cuda")
        model.cuda()

    if os.path.isfile(_modelpath):
        print("=> loading checkpoint '{}'".format(_modelpath))
        checkpoint = torch.load(_modelpath)

        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(_modelpath, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'")
        exit()
    print(model)

    return model

def data_preprocess(batchsize=64, test_batch_size=256, usegpu=True, seed=1, DATASET='cifar10'):
    torch.manual_seed(seed)
    memorysetting = False
    if cuda and usegpu:
        torch.cuda.manual_seed(seed)
        memorysetting = True
    DATASET = DATASET

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.' + DATASET, train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=batchsize, shuffle=True, pin_memory=memorysetting)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.' + DATASET, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=test_batch_size, shuffle=True, pin_memory=memorysetting)

    return train_loader, test_loader


def iterative_pruning(save_path, dataset='cifar10', net='resnet', _MODELPATH='', pretrained=False, start_epoch=0, epochs=150, session_len=20, percentile=0.6, lr=0.1, momentum=0.9, weightdecay=1e-4, depth=19, sparsity=0.0001, sparsity_regulation=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = models.__dict__[net](dataset=dataset, depth=depth)

    if pretrained:
        if os.path.isfile(_MODELPATH):
            checkpoint = torch.load(_MODELPATH)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])#
            print("loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                .format(_MODELPATH, checkpoint['epoch'], best_prec1))
        else:
            print("no checkpoint found at '{}'".format(_MODELPATH))

    if cuda:
        model.cuda()

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            loss.backward()
            if sparsity_regulation:
                utils.updateBN(model, sparsity)

            # Prevent pruned weight from updating
            for layer in model.modules():
                if isinstance(layer, nn.Conv2d):
                    if layer.__hash__() in mask:
                        layer.weight.grad.data.mul_(mask[layer.__hash__()])

            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data))

    def unstructured_pruning(percentile):
        # sample = True
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                total = layer.weight.data.numel()
                weight_matrix = layer.weight.data.view(-1).abs().clone()

                sorted_weight, _ = torch.sort(weight_matrix)

                threshold = sorted_weight[int(np.ceil(total * percentile))]
                m_weight = layer.weight.data.abs().clone()

                mask[layer.__hash__()] = m = m_weight.gt(threshold).float().cuda()

                # Restore to original initialization
                layer.weight.data = initial_weight[layer.__hash__()].clone()

                layer.weight.data.mul_(m)
                # if sample:
                #     print(layer.weight.data)
                #     sample = False

    best_prec1 = 0.
    iteration_percentile = change_factor = (1 - percentile) ** (1 / np.floor(epochs / session_len))
    initial_weight = {}
    mask = {}
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            initial_weight[layer.__hash__()] = layer.weight.data.clone()

    for epoch in range(start_epoch, epochs):
        # print(iteration_percentile)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weightdecay)
        if epoch != 0 and epoch % session_len == 0:
            print("--Pruned: ", 1 - iteration_percentile, "\n")
            unstructured_pruning(1 - iteration_percentile)
            iteration_percentile *= change_factor
            # Reset best
            best_prec1 = 0.
        if epoch in [epochs * 0.5, epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train(epoch)
        prec1 = test(model)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath=save_path)
        # for layer in model.modules():
        #     if isinstance(layer, nn.Conv2d):
        #         print(layer.weight.data)
        #         break

    print("Best accuracy: " + str(best_prec1))


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction = 'sum').data # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


# Prepare and train over-parameterized
train_loader, test_loader = data_preprocess()
iterative_pruning('./iterative', net='resnet', epochs=159, depth=164, session_len=20, percentile=0.4)

# model = load_model("./iterative/checkpoint.pth.tar", net='resnet', depth=164)

# test(model)