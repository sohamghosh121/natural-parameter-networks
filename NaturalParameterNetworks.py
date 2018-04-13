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

class GaussianNPNCrossEntropy(autograd.Function):
    """
        Implements KL Divergence loss for output layer
        KL(N(o_c, o_d) || N(y_m, diag(eps))
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
    """
    @staticmethod
    def forward(ctx, o_c, o_d, y, eps):
        ctx.save_for_backward(o_c, o_d, y)
        k = torch.Tensor([y.size()[0]])
        det_ratio = torch.Tensor([torch.prod(o_d / eps)])
        print('o_d', torch.prod(o_d / eps))
        KL = 0.5 * (torch.sum(o_d/eps) + torch.sum(torch.pow(o_c - y, 2) / eps) - k + torch.log(det_ratio))
        return KL

    @staticmethod
    def backward(ctx, grad_output):
        o_c, o_d, y = ctx.saved_variables
        o_m, o_s = gaussian_f(o_c, o_d)
        grad__o_m = grad__o_s = None
        grad__o_m = (torch.sigmoid(kappa(o_s) * o_m) - y) * kappa(o_s)
        grad__o_s = (torch.sigmoid(kappa(o_s) * o_m) - y) * o_m * (- PI / 16.0 * torch.pow(1 + PI * o_s / 8.0 , -1.5))
        return grad__o_m, grad__o_s, None, None
        

class GaussianNPNLayer(nn.Module):
    def __init__(self, input_features, output_features, activation=None):
        super(GaussianNPNLayer, self).__init__()
        self.weight_c = nn.Parameter(torch.Tensor(input_features, output_features))
        self.weight_d = nn.Parameter(torch.Tensor(input_features, output_features))
        self.bias_c = nn.Parameter(torch.Tensor(output_features))
        self.bias_d = nn.Parameter(torch.Tensor(output_features))

        # TODO: check how to do initialisation
        self.weight_c.data.uniform_(-1.0, 1.0)
        self.weight_d.data.uniform_(-1., 0.0)
        self.bias_c.data.uniform_(-1.0, 1.)
        self.bias_d.data.uniform_(-1., 0.0)
        self.activation = activation

    def forward(self, input):
        input_m, input_s = input
        # input_m, input_s = gaussian_f(input_c, input_d)
        weight_m, weight_s = gaussian_f(self.weight_c, self.weight_d)
        bias_m, bias_s = gaussian_f(self.bias_c, self.bias_d)
        o_m = torch.matmul(input_m, weight_m)
        o_m += bias_m.unsqueeze(0).expand_as(o_m)
        o_s = torch.matmul(input_s, weight_s) + \
            torch.matmul(input_s, torch.pow(weight_m, 2)) + \
            torch.matmul(torch.pow(input_s, 2), weight_s)
        o_s += bias_s.unsqueeze(0).expand_as(o_s)
        o_c, o_d = gaussian_f_inv(o_m, o_s)
        if self.activation == 'sigmoid':
            a_m = torch.sigmoid(o_c / torch.sqrt(1 + ETA * o_d))
            a_s = torch.sigmoid((ALPHA * (o_c + BETA))/torch.sqrt(1 + ETA * ALPHA ** 2.0 * o_d)) - (a_m ** 2)
            return a_m, a_s
        else:
            return o_c, o_d


class GaussianNPN(nn.Module):
    def __init__(self, input_features, output_classes, hidden_sizes, activation='sigmoid', eps=0.5):
        super(GaussianNPN, self).__init__()
        assert(len(hidden_sizes) >= 0)
        self.num_classes = output_classes
        self.layers = []
        for i, h_sz in enumerate(hidden_sizes):
            if i == 0 :
                h = GaussianNPNLayer(input_features, hidden_sizes[i], activation)
            else:
                h = GaussianNPNLayer(hidden_sizes[i-1], hidden_sizes[i], activation)
            self.layers.append(h)
        if len(hidden_sizes) > 0:
            self.layers.append(GaussianNPNLayer(hidden_sizes[-1], output_classes))
        self.epsilon = torch.ones(output_classes) * eps
        self.net = nn.Sequential(*list(self.layers)) # just to make model.parameters() work

    def forward(self, x_m, x_s):
        return self.net((x_m, x_s))

    def loss(self, input, y):
        """
            output - torch.LongTensor
        """
        out = input
        for ix, L in enumerate(self.layers):
            out = L(out)
        
        o_c, o_d = out
        loss = GaussianNPNCrossEntropy.apply(o_c, o_d, y, self.epsilon)
        return loss
