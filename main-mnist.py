"""
    Tests NPN
    Borrowed from https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import datasets
from NaturalParameterNetworks import GaussianNPN

from torchviz import make_dot, make_dot_from_trace

SEED = 42  # my favorite seed

torch.manual_seed(SEED)
np.random.seed(SEED)

def train(model, optimizer, epoch, batch_size=128, log_interval=10):
    model.train()
    train_loss = 0
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.view(data.size(0), -1)
        data_m = Variable(data)
        data_s = Variable(torch.zeros(data_m.size()))
        trace, _ = torch.jit.trace(model, args=( data_m,data_s ))
        target_onehot = torch.FloatTensor(data.size(0), 10)
        target_onehot.zero_()
        target_onehot.scatter_(1, target.unsqueeze(1), 1)
        target_onehot = Variable(target_onehot)

        loss = model.loss(data_m, data_s, target_onehot)

        loss.backward()

        optimizer.step()
        train_loss += loss.data.numpy()[0]
        
        output = model(data_m, data_s)
        pred = output[0].data.max(1, keepdim=True)[1] # get the index of the max log-probability
        train_correct += pred.eq(target.view_as(pred)).long().cpu().sum()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss , train_correct, len(train_loader.dataset),
        100. * train_correct / len(train_loader.dataset)))
    

def test(model, batch_size=128):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(data.size(0), -1)
        data_m = Variable(data)
        data_s = Variable(torch.zeros(data_m.size()))
        target_onehot = torch.FloatTensor(data.size(0), 10)
        target_onehot.zero_()
        target_onehot.scatter_(1, target.unsqueeze(1), 1)
        target_onehot = Variable(target_onehot)
        output = model(data_m, data_s)
        
        loss = model.loss(data_m, data_s, target_onehot)
        pred = output[0].data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        test_loss += loss.data.numpy()[0]

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

if __name__ == '__main__':
    BATCH_SZ = 128

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=BATCH_SZ, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=BATCH_SZ, shuffle=True)

    model = GaussianNPN(784, 10, [400,400])
    optimizer = optim.Adadelta(model.parameters())

    for epoch in range(1, 100 + 1):
        train(model, optimizer, epoch, batch_size=BATCH_SZ)
        test(model)
