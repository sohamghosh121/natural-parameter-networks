"""
    Defines implementation for NPNs
"""

import torch 
import torch.nn as nn
import torch.autograd as autograd
import torch.Fun


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

        f_inv(m, s) ==> N(-c/2d, -1/2d)
    """
    c = m/s
    d = -1.0/(2.0 * s)
    return (c, d)

class GaussianNPNFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input_statistics, weight_statistics, bias_statistics):
        # compute linear pass
        ctx.save_for_backward(input_statistics, weight_statistics, bias_statistics)
        # compute non-linear activation

        (input_c, input_d) = input_statistics
        (weight_c, weight_d) = weight_statistics
        (bias_c, bias_d) = bias_statistics

        input_m, input_s = gaussian_f(input_c, input_d)
        weight_m, weight_s = gaussian_f(weight_c, weight_d)
        bias_m, bias_s = gaussian_f(bias_c, bias_d)

        output_m = torch.mm(weight_m.t(), input_m) + bias_m
        output_s = torch.mm(weight_s.t(), input_s) + bias_s

        output_c, output_d = gaussian_f_inv(output_m, output_s)
        return (output_c, output_d)

    @staticmethod
    def backward(ctx, grad_output):
        (input_c, input_d), (weight_c, weight_d), (bias_c, bias_d) = ctx.saved_variables
        grad_weight_c = grad_weight_d = grad_bias_c = grad_bias_d = None
        grad_input = grad_output = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        # if ctx.needs_input_grad[0]:
        #     grad_input = grad_output.mm(weight_c)
        # if ctx.needs_input_grad[1]:
        #     grad_weight = grad_output.t().mm(input)
        # if bias is not None and ctx.needs_input_grad[2]:
        #     grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, (grad_weight_c, grad_weight_d), (grad_bias_c, grad_bias_d)



class GaussianNPN(nn.Module):
    def __init__(self, input_features, output_features):
        self.weight_c = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight_d = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias_c = nn.Parameter(torch.Tensor(output_features))
        self.bias_d = nn.Parameter(torch.Tensor(output_features))

        # TODO: check how to do initialisation
        self.weight_c.data.uniform_(-0.1, 0.1)
        self.weight_d.data.uniform_(-0.1, 0.1)
        self.bias_c.data.uniform_(-0.1, 0.1)
        self.bias_d.data.uniform_(-0.1, 0.1)

        

    def forward(self, input):
        GaussianNPNFunction.apply(input,
            (self.weight_c, self.weight_d),
            (self.bias_c, self.bias_d))