"""
    Defines implementation for NPNs
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as O
import torch.autograd as autograd
import torch.nn.functional as F

PI = float(np.pi)
ETA_SQ = float(np.pi / 8.0)
ALPHA = float(4.0 - 2.0 * np.sqrt(2))
ALPHA_SQ = float(ALPHA ** 2.0)
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
    return 1 / torch.sqrt(1 + x * ETA_SQ)

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
        KL = (torch.sum(o_s/eps, 1) + torch.sum(torch.pow(o_m - y, 2) / eps, 1) - k + torch.log(det_ratio)) * 0.5
        return torch.Tensor([torch.mean(KL)])
        return torch.mean
        

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError()
        

class GaussianNPNLinearLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(GaussianNPNLinearLayer, self).__init__()
        self.W_m, self.M_s = self.get_init_W(input_features, output_features)
        self.b_m, self.p_s = self.get_init_b(output_features)

    def get_init_W(self, _in, _out):
        # obtained from the original paper
        W_m = 2 * np.sqrt(6)/ np.sqrt(_in + _out) * (np.random.rand(_in, _out) - 0.5)
        W_s = 1 * np.sqrt(6)/ np.sqrt(_in + _out) * (np.random.rand(_in, _out))
        M_s = np.log(np.exp(W_s) - 1)
        return nn.Parameter(torch.FloatTensor(W_m)), nn.Parameter(torch.FloatTensor(M_s))

    def get_init_b(self, _out):
        b_m = np.zeros((_out))
        # instead of bias_s, parametrize as log(1+exp(pias_s))
        p_s = np.exp(-1 * np.ones((_out)))
        return nn.Parameter(torch.Tensor(b_m)), nn.Parameter(torch.Tensor(p_s))

    def forward(self, input_m, input_s):
        # do this to ensure positivity of W_s, b_s
        b_s = torch.log(1 + torch.exp(self.p_s))
        W_s = torch.log(1 + torch.exp(self.M_s))

        o_m = torch.addmm(self.b_m, input_m, self.W_m)
        o_s = b_s + torch.mm(input_s, W_s) + \
            torch.mm(input_s, torch.pow(self.W_m, 2)) + \
            torch.mm(torch.pow(input_m, 2), W_s)
        return o_m, o_s


class GaussianNPNNonLinearity(nn.Module):
    def __init__(self, activation):
        super(GaussianNPNNonLinearity, self).__init__()
        self.activation = activation

    def forward(self, o_m, o_s):
        if self.activation == 'sigmoid':
            a_m = torch.sigmoid(o_m * kappa(o_s))
            a_s = torch.sigmoid(((o_m + BETA) * ALPHA)/torch.sqrt(1 + o_s * ETA_SQ * ALPHA_SQ)) - torch.pow(a_m, 2)
            return a_m, a_s
        else:
            return o_m, o_s


class GaussianNPN(nn.Module):
    def __init__(self, input_features, output_classes, hidden_sizes, activation='sigmoid', eps=0.5):
        super(GaussianNPN, self).__init__()
        assert(len(hidden_sizes) >= 0)
        self.num_classes = output_classes
        self.layers = []
        self.params = []
        for i, h_sz in enumerate(hidden_sizes):
            if i == 0 :
                h = GaussianNPNLinearLayer(input_features, hidden_sizes[i])
            else:
                h = GaussianNPNLinearLayer(hidden_sizes[i-1], hidden_sizes[i])
            self.layers.append(h)
            self.layers.append(GaussianNPNNonLinearity(activation))
        if len(hidden_sizes) > 0:
            self.layers.append(GaussianNPNLinearLayer(hidden_sizes[-1], output_classes))
        else:
            self.layers.append(GaussianNPNLinearLayer(input_features, output_classes))

        self.layers.append(GaussianNPNNonLinearity(activation))
        self.net = nn.Sequential(*list(self.layers)) # just to make model.parameters() work
        self.lossfn = nn.BCELoss()

    def forward(self, a_m, a_s):
        for L in self.layers:
            a_m, a_s = L(a_m, a_s)
        return a_m, a_s

    def loss(self, x_m, x_s, y):
        """
            output - torch.LongTensor
        """
        # print(id(x_m))
        a_m, a_s = self.forward(x_m, x_s)
        return self.lossfn(a_m, y)
