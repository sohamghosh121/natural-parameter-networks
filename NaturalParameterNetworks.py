"""
    Defines implementation for NPNs
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as O
import torch.autograd as autograd
import torch.nn.functional as F

PI = np.pi
ETA = float(np.pi / 8.0)
ALPHA = float(4.0 - 2.0 * np.sqrt(2))
BETA = - float(np.log(np.sqrt(2) + 1))

def gaussian_f(c, d):
    """
        Calculates mean and variance given natural parameters
        >> m, s = f(c, d)

        f(c, d) ==> N(-c/2d, -1/2d)
    """
    m = -c/(2.0 * d)
    s = -1.0/(2.0 * d)
    return (m, s)


def gaussian_f_inv(m, s):
    """
        Calculates natural parameters given mean, variance
        >> c, d = f_inv(m, s)

        f_inv(m, s) ==> exp(m/s, -1/2s)
    """
    c = m/s
    d = -1.0/(2.0 * s)
    return (c, d)

def kappa(x):
    return torch.pow(1 + PI/8 * x, -0.5)

class GaussianNPNKLDivergence(autograd.Function):
    """
        Implements KL Divergence loss for output layer
        KL(N(o_m, o_s) || N(y_m, diag(eps))
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
    """
    @staticmethod
    def forward(ctx, o_m, o_s, y, eps):
        ctx.save_for_backward(o_m, o_s, y)
        k = torch.Tensor([y.size(1)])
        det_ratio = torch.prod(o_s / eps, 1)
        KL = 0.5 * (torch.sum(o_s/eps, 1) + torch.sum(torch.pow(o_m - y, 2) / eps, 1) - k + torch.log(det_ratio))
        return torch.Tensor([torch.mean(KL)])
        return torch.mean
        

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError()
        

class GaussianNPNLinearLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(GaussianNPNLinearLayer, self).__init__()
        self.weight_m, self.weight_s = self.get_init_W(input_features, output_features)
        self.bias_m, self.bias_s = self.get_init_b(output_features)

    def get_init_W(self, _in, _out):
        # obtained from the original paper
        W_m = 2 * np.sqrt(6)/ np.sqrt(_in + _out) * (np.random.rand(_in, _out) - 0.5)
        W_s = 1 * np.sqrt(6)/ np.sqrt(_in + _out) * (np.random.rand(_in, _out))
        return nn.Parameter(torch.FloatTensor(W_m)), nn.Parameter(torch.FloatTensor(W_s))

    def get_init_b(self, _out):
        b_m = np.zeros((_out))
        b_s = np.log(np.exp(-1 * np.ones((_out)) ) + 1)
        return nn.Parameter(torch.Tensor(b_m)), nn.Parameter(torch.Tensor(b_s))

    def forward(self, input, activation=None):
        input_m, input_s = input
        o_m = torch.addmm(self.bias_m, input_m, self.weight_m)
        o_s = self.bias_s + torch.mm(input_s, self.weight_s) + \
            torch.mm(input_s, torch.pow(self.weight_m, 2)) + \
            torch.mm(torch.pow(input_m, 2), self.weight_s)
        return o_m, o_s


class GaussianNPNNonLinearity(nn.Module):
    def __init__(self, activation):
        super(GaussianNPNNonLinearity, self).__init__()
        self.activation = activation

    def forward(self, input):
        o_m, o_s = input    
        if self.activation == 'sigmoid':
            a_m = torch.sigmoid(o_m / torch.sqrt(1 + ETA * o_s))
            a_s = torch.sigmoid((ALPHA * (o_m + BETA))/torch.sqrt(1 + ETA * ALPHA ** 2.0 * o_s)) - (a_m ** 2)
            return a_m, a_s
        else:
            return o_m, o_s


class GaussianNPN(nn.Module):
    def __init__(self, input_features, output_classes, hidden_sizes, activation='sigmoid', eps=0.5):
        super(GaussianNPN, self).__init__()
        assert(len(hidden_sizes) >= 0)
        self.num_classes = output_classes
        self.layers = []
        for i, h_sz in enumerate(hidden_sizes):
            if i == 0 :
                h = GaussianNPNLinearLayer(input_features, hidden_sizes[i])
            else:
                h = GaussianNPNLinearLayer(hidden_sizes[i-1], hidden_sizes[i])
            self.layers.append(GaussianNPNNonLinearity(activation))
            self.layers.append(h)
        if len(hidden_sizes) > 0:
            self.layers.append(GaussianNPNLinearLayer(hidden_sizes[-1], output_classes))
        else:
            self.layers.append(GaussianNPNLinearLayer(input_features, output_classes))
        self.layers.append(GaussianNPNNonLinearity(activation))
        self.net = nn.Sequential(*list(self.layers)) # just to make model.parameters() work
        self.lossfn = nn.BCELoss(size_average=False)

    def forward(self, x_m, x_s):
        return self.net((x_m, x_s))

    def loss(self, input, y):
        """
            output - torch.LongTensor
        """
        a_m, a_s = self.net(input)
        return self.lossfn(a_m, y)
