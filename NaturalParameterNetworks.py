"""
    Defines implementation for NPNs
"""

import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


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
        >> c, d = f_inv(c, d)

        f_inv(m, s) ==> N(m/s, -1/2s)
    """
    c = m/s
    d = -1.0/(2.0 * s)
    return (c, d)

class GaussianNPNFunction(autograd.Function):
    """
        NPN Function that takes previous layer outputs
        and does forward/backward pass
    """
    @staticmethod
    def forward(ctx, input_c, input_d, weight_c, weight_d, bias_c, bias_d):
        # compute linear pass
        ctx.save_for_backward(input_c, input_d, weight_c, weight_d, bias_c, bias_d)
        input_m, input_s = gaussian_f(input_c, input_d)
        weight_m, weight_s = gaussian_f(weight_c, weight_d)
        bias_m, bias_s = gaussian_f(bias_c, bias_d)
        output_m = torch.matmul(weight_m.t(), input_m) + bias_m
        output_s = torch.mm(weight_s.t(), input_s) + \
            torch.mm(torch.pow(weight_m, 2).t(), input_s) + \
            torch.mm(weight_s.t(), torch.pow(input_s, 2)) + \
            bias_s
        output_c, output_d = gaussian_f_inv(output_m, output_s)
        return (output_c, output_d)

    @staticmethod
    def backward(ctx, grad_output):
        input_c, input_d, weight_c, weight_d, bias_c, bias_d = ctx.saved_variables
        # calculate gradients here
        grad_weight_c = grad_weight_d = grad_bias_c = grad_bias_d = None
        grad_input = grad_output = None

        return grad_input, grad_weight_c, grad_weight_d, grad_bias_c, grad_bias_d


class GaussianNPNSigmoidFunction(autograd.Function):
    """
        Sigmoid Activation Function for Gaussian NPN
    """
    @staticmethod
    def forward(ctx, output_c, output_d):
        # compute non-linear activation
        activation_m = torch.sigmoid(output_c / torch.sqrt(1 + ETA * output_d))
        activation_s = torch.sigmoid((ALPHA * (output_c + BETA))/torch.sqrt(1 + ETA * ALPHA ** 2.0 * output_d)) - (activation_m ** 2)
        return (activation_m, activation_s)

    def backward(ctx, grad_output):
        output_c, output_d = ctx.saved_variables
        # calculate gradients here
        grad_input = grad_output = None
        return grad_input


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
        # ctx.save_for_backward()
        k = torch.Tensor([y.size()[0]])
        det_ratio = torch.Tensor([torch.prod(o_d / eps)])
        KL = 0.5 * (torch.sum(o_d/eps) + torch.sum(torch.pow(o_c - y, 2) / eps) - k + torch.log(det_ratio))
        return KL

    @staticmethod
    def backward(ctx, grad_output):
        o_c, o_d, y = ctx.saved_variables
        o_m, o_s = gaussian_f(o_c, o_d)
        grad__o_m = grad__o_s = None
        grad__o_m = (torch.sigmoid(kappa(o_s) * o_m) - y) * kappa(o_s)
        grad__o_s = (torch.sigmoid(kappa(o_s) * o_m) - y) * o_m * (- PI / 16.0 * torch.pow(1 + PI * o_s / 8.0 , -1.5))
        return grad__o_m, grad__o_s
        

class GaussianNPNLayer(nn.Module):
    def __init__(self, input_features, output_features, activation=None):
        super(GaussianNPNLayer, self).__init__()
        self.weight_c = nn.Parameter(torch.Tensor(input_features, output_features))
        self.weight_d = nn.Parameter(torch.Tensor(input_features, output_features))
        self.bias_c = nn.Parameter(torch.Tensor(output_features, 1))
        self.bias_d = nn.Parameter(torch.Tensor(output_features, 1))

        # TODO: check how to do initialisation
        self.weight_c.data.uniform_(-0.1, 0.1)
        self.weight_d.data.uniform_(-0.1, 0.1)
        self.bias_c.data.uniform_(-0.1, 0.1)
        self.bias_d.data.uniform_(-0.1, 0.1)

        self.activation = activation

    def forward(self, input):
        input_c, input_d = input
        pre_activation = GaussianNPNFunction.apply(input_c, input_d,
            self.weight_c, self.weight_d,
            self.bias_c, self.bias_d)
        if self.activation == 'sigmoid':
            o_c, o_d = pre_activation
            return GaussianNPNSigmoidFunction.apply(o_c, o_d)
        else:
            return pre_activation


class GaussianNPN(nn.Module):
    def __init__(self, input_features, output_classes, hidden_sizes, activation='sigmoid', eps=0.1):
        super(GaussianNPN, self).__init__()
        assert(len(hidden_sizes) >= 0)
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


    def forward(self, input):
        out = input
        for L in self.layers:
            out = L(out)
        return out

    def loss(self, input, output):
        out = input
        for L in self.layers:
            out = L(out)
        o_c, o_d = out
        loss = GaussianNPNCrossEntropy.apply(o_c, o_d, output, self.epsilon)
        return loss


if __name__ == '__main__':
    # do a small test
    g = GaussianNPN(10, 2, [5,5])
    m = autograd.Variable(torch.FloatTensor(np.random.rand(10, 1)))
    s = autograd.Variable(torch.FloatTensor(np.random.rand(10, 1)))
    y = torch.Tensor([[1,0]]).t()
    loss = g.loss((m, s), y)
    print(loss)

