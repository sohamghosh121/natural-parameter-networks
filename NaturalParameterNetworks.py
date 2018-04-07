"""
    Defines implementation for NPNs
"""

import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


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
        >> c, d = f_inv(c, d)

        f_inv(m, s) ==> N(m/s, -1/2s)
    """
    c = m/s
    d = -1.0/(2.0 * s)
    return (c, d)

class GaussianNPNFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input_c, input_d, weight_c, weight_d, bias_c, bias_d):
        # compute linear pass
        ETA = float(np.pi / 8.0)
        ALPHA = float(4.0 - 2.0 * np.sqrt(2))
        BETA = - float(np.log(np.sqrt(2) + 1))

        ctx.save_for_backward(input_c, input_d, weight_c, weight_d, bias_c, bias_d)
        # compute non-linear activation

        input_m, input_s = gaussian_f(input_c, input_d)
        weight_m, weight_s = gaussian_f(weight_c, weight_d)
        bias_m, bias_s = gaussian_f(bias_c, bias_d)

        output_m = torch.mm(weight_m.t(), input_m) + bias_m
        output_s = torch.mm(weight_s.t(), input_s) + bias_s

        output_c, output_d = gaussian_f_inv(output_m, output_s)

        # non-linearity
        # TODO: implement other activation functions
        activation_m = torch.sigmoid(output_c / torch.sqrt(1 + ETA * output_d))
        activation_s = torch.sigmoid((ALPHA * (output_c + BETA))/torch.sqrt(1 + ETA * ALPHA ** 2.0 * output_d)) - (activation_m ** 2)
        
        return (activation_m, activation_s)

    @staticmethod
    def backward(ctx, grad_output):
        input_c, input_d, weight_c, weight_d, bias_c, bias_d = ctx.saved_variables

        # calculate gradients here
        grad_weight_c = grad_weight_d = grad_bias_c = grad_bias_d = None
        grad_input = grad_output = None

        return grad_input, grad_weight_c, grad_weight_d, grad_bias_c, grad_bias_d

class GaussianNPNLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(GaussianNPNLayer, self).__init__()
        self.weight_c = nn.Parameter(torch.Tensor(input_features, output_features))
        self.weight_d = nn.Parameter(torch.Tensor(input_features, output_features))
        self.bias_c = nn.Parameter(torch.Tensor(output_features))
        self.bias_d = nn.Parameter(torch.Tensor(output_features))

        # TODO: check how to do initialisation
        self.weight_c.data.uniform_(-0.1, 0.1)
        self.weight_d.data.uniform_(-0.1, 0.1)
        self.bias_c.data.uniform_(-0.1, 0.1)
        self.bias_d.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        input_c, input_d = input
        GaussianNPNFunction.apply(input_c, input_d,
            self.weight_c, self.weight_d,
            self.bias_c, self.bias_d)


class CrossEntropyLossGaussianNPN(autograd.Function):
    def forward(self, input, target):
        # TODO: now using default parameters - what do they mean?
        input_c, input_d = input
        input_m, input_s = gaussian_f(input_c, input_d)
        return F.cross_entropy(input_m, target, self.weight, True,
                               -100, True)

    def kappa(self, x):
        ETA = np.pi / 8.0
        return torch.sqrt(1 + ETA * x)

    def backward(self, grad_output):
        pass


class GaussianNPN(nn.Module):
    def __init__(self, input_features, output_classes, hidden_sizes):
        super(GaussianNPN, self).__init__()
        assert(len(hidden_sizes) >= 0)
        self.layers = []
        for i, h_sz in enumerate(hidden_sizes):
            if i == 0 :
                h = GaussianNPNLayer(input_features, hidden_sizes[i])
            else:
                h = GaussianNPNLayer(hidden_sizes[i-1], hidden_sizes[i])
            self.layers.append(h)
        if len(hidden_sizes) > 0:
            self.layers.append(GaussianNPNLayer(hidden_sizes[-1], hidden_sizes[i]))            

    def forward(self, input):
        out = input
        for L in self.layers:
            out = L(out)
            print(out)
        return out
        

if __name__ == '__main__':
    g = GaussianNPN(10, 5, [2, 2])
    m = autograd.Variable(torch.FloatTensor(np.random.rand(10, 1)))
    s = autograd.Variable(torch.FloatTensor(np.random.rand(10, 1)))
