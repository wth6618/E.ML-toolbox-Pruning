import torch
from utils import *
import models

from unstructured_weight import Unstructured
from channel_selection import Channel_selection
from soft_filter import Soft_filter
from structured_l1 import L1

cuda = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def test(model, dataset='cifar10'):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


def train(model, optimizer, epoch):
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
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data))


def adjust_learning_rate(optimizer, epoch, lr):
    if epoch in [41, 42]:
        lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def refine(model, savepath, epochs=40, lr=0.001):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    best_acc = 0
    # Train and val
    for epoch in range(0, epochs):

        adjust_learning_rate(optimizer, epoch, lr)

        train(model, optimizer, epoch)
        test_acc = test(model)


        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, savepath)

    print('Best acc:')
    print(best_acc)


train_loader, test_loader = DataPreprocess()

PreTrain('./50', Net='resnet', depth=50, epochs=120)

# model = LoadModel('./resnet_56_cifar10/model_best.pth.tar', Net='resnet', depth=56, refine=False, usegpu=False)
#
# # test(model)
#
# engine = L1(model, pruning_rate=0.1)
# engine.step()
# engine.zero_params()
#
# test(model)
#
# # refine(model, './resnet_56_cifar10/a')
#
# test(model)
