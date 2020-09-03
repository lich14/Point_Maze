import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
from Utils import LayerNorm, fanin_init, zeros, ones
from torch.distributions import Categorical, Distribution, Normal


def identity(x):
    return x


class MLP_Net(nn.Module):

    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=torch.relu,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=True,
            output_activation=identity,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.layer_norm = layer_norm

        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for _, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.layer_norms.append(ln)

        self.fcs = nn.ModuleList(self.fcs)
        self.layer_norms = nn.ModuleList(self.layer_norms)

        self.last_fc = nn.Linear(in_size, self.output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)

        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        return output


class Plus_Net(MLP_Net):

    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            option_num,
            init_w=3e-3,
            hidden_activation=torch.relu,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=True,
            output_activation=identity,
    ):

        super().__init__(
            hidden_sizes,
            output_size,
            input_size + option_num,
            init_w,
            hidden_activation,
            hidden_init,
            b_init_value,
            layer_norm,
            output_activation,
        )

        self.option_num = option_num

    def forward(self, input, option):
        option = torch.nn.functional.one_hot(option.long(), self.option_num).squeeze()
        flat_inputs = torch.cat([input, option.double()], dim=1)
        return super().forward(flat_inputs)


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean + self.normal_std *
            Normal(zeros(self.normal_mean.size()).type_as(self.normal_mean),
                   ones(self.normal_std.size()).type_as(self.normal_mean)).sample())
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)