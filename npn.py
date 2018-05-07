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
# for sigmoid
ALPHA = float(4.0 - 2.0 * np.sqrt(2))
ALPHA_SQ = float(ALPHA ** 2.0)
BETA = - float(np.log(np.sqrt(2) + 1))
# for tanh
ALPHA_2 = float(8.0 - 4.0 * np.sqrt(2))
ALPHA_2_SQ = float(ALPHA ** 2.0)
BETA_2 = - float(0.5 * np.log(np.sqrt(2) + 1))

def kappa(x, const=1.0, alphasq= 1.0):
    return 1 / torch.sqrt(const + x * alphasq * ETA_SQ)

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

    def forward(self, input):
        if isinstance(input, tuple):
            input_m, input_s = input
        elif isinstance(input, torch.Tensor):
            input_m = input
            input_s = autograd.Variable(input.new().zero_())
        else:
            raise ValueError('input was not a tuple or torch.Tensor (%s)' % type(input))

        # do this to ensure positivity of W_s, b_s
        b_s = torch.log(1 + torch.exp(self.p_s))
        W_s = torch.log(1 + torch.exp(self.M_s))

        o_m = self.b_m + torch.matmul(input_m, self.W_m)
        o_s = b_s + torch.matmul(input_s, W_s) + \
            torch.matmul(input_s, torch.pow(self.W_m, 2)) + \
            torch.matmul(torch.pow(input_m, 2), W_s)
        return o_m, o_s


class GaussianNPNNonLinearity(nn.Module):
    # TODO: does it help to define this in a function instead?
    def __init__(self, activation):
        super(GaussianNPNNonLinearity, self).__init__()
        self.activation = activation

    def forward(self, o):
        o_m, o_s = o
        if self.activation == 'sigmoid':
            a_m = torch.sigmoid(o_m * kappa(o_s))
            a_s = torch.sigmoid(((o_m + BETA) * ALPHA) * kappa(o_s, alphasq=ALPHA_SQ)) - torch.pow(a_m, 2)
            return a_m, a_s
        elif self.activation == 'tanh':
            a_m = 2.0 * torch.sigmoid(o_m * kappa(o_s, const=0.25)) - 1
            a_s = 4.0 * torch.sigmoid(((o_m + BETA_2) * ALPHA_2) * kappa(o_s, alphasq=ALPHA_2_SQ)) - torch.pow(a_m, 2) - 2.0 * a_m - 1.0
            return a_m, a_s
        else:
            return o_m, o_s


class GaussianNPN(nn.Module):
    def __init__(self, input_features, output_classes, hidden_sizes, activation='sigmoid'):
        super(GaussianNPN, self).__init__()
        assert(len(hidden_sizes) >= 0)
        self.num_classes = output_classes
        self.layers = []
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

        self.layers.append(GaussianNPNNonLinearity('sigmoid')) # last one needs to be sigmoid
        self.net = nn.Sequential(*list(self.layers)) # just to make model.parameters() work
        self.lossfn = nn.BCELoss(size_average=False)

    def forward(self, a):
        for L in self.layers:
            a = L(a)
        a_m, a_s = a
        return a_m, a_s

    def loss(self, x_m, x_s, y):
        a_m, a_s = self.forward((x_m, x_s))
        return self.lossfn(a_m, y)
