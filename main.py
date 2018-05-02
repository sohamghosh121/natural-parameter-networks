"""
    Tests NPN
    Borrowed from https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import datasets
from NaturalParameterNetworks import GaussianNPN

def train(model, optimizer, epoch, batch_size=32, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batch_size, 784)
        data_m = Variable(data)
        data_s = Variable(torch.zeros(data_m.size()))
        target_onehot = torch.FloatTensor(batch_size, 10)
        target_onehot.zero_()
        target_onehot.scatter_(1, target.unsqueeze(1), 1)
        target_onehot = Variable(target_onehot)

        optimizer.zero_grad()
        loss = model.loss((data_m, data_s), target_onehot)
        loss.backward()
        optimizer.step()
        # exit()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    

def test(model, batch_size=32):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(batch_size, 784).t()
        data_m = Variable(data)
        data_s = Variable(torch.zeros(data_m.size()))
        target_onehot = torch.FloatTensor(32, 10)
        target_onehot.zero_()
        target_onehot.scatter_(1, target.unsqueeze(1), 1)
        target_onehot = Variable(target_onehot)
        output = model(data_m, data_s)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

if __name__ == '__main__':
    BATCH_SZ = 32

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=BATCH_SZ, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=BATCH_SZ, shuffle=True)

    model = GaussianNPN(784, 10, [128])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(1, 100 + 1):
        train(model, optimizer, epoch, batch_size=BATCH_SZ)
        test(model)
