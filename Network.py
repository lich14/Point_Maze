import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import LayerNorm, zeros, ones
from torch.distributions import Categorical, Distribution, Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class PopArt(nn.Module):

    def __init__(self, output_layer, beta: float = 0.0001, zero_debias: bool = False, start_pop: int = 0):
        # zero_debias=True and start_pop=8 seem to improve things a little but (False, 0) works as well
        super(PopArt, self).__init__()
        self.start_pop = start_pop
        self.zero_debias = zero_debias
        self.beta = beta
        self.output_layers = output_layer if isinstance(output_layer, (tuple, list, nn.ModuleList)) else (output_layer,)
        #shape = self.output_layers[0].bias.shape
        shape = 1
        device = self.output_layers[0].bias.device
        #assert all(shape == x.bias.shape for x in self.output_layers)
        self.mean = nn.Parameter(torch.zeros(shape, device=device), requires_grad=False)
        self.mean_square = nn.Parameter(torch.ones(shape, device=device), requires_grad=False)
        self.std = nn.Parameter(torch.ones(shape, device=device), requires_grad=False)
        self.updates = 0

    def forward(self, *input):
        pass

    @torch.no_grad()
    def update(self, targets):
        targets_shape = targets.shape
        targets = targets.view(-1, 1)
        beta = max(1. / (self.updates + 1.), self.beta) if self.zero_debias else self.beta
        # note that for beta = 1/self.updates the resulting mean, std would be the true mean and std over all past data
        new_mean = (1. - beta) * self.mean + beta * targets.mean(0)
        new_mean_square = (1. - beta) * self.mean_square + beta * (targets * targets).mean(0)
        new_std = (new_mean_square - new_mean * new_mean).sqrt().clamp(0.0001, 1e6)
        assert self.std.shape == (1,), 'this has only been tested in 1D'
        if self.updates >= self.start_pop:
            for layer in self.output_layers:
                layer.weight *= self.std / new_std
                layer.bias *= self.std
                layer.bias += self.mean - new_mean
                layer.bias /= new_std
        self.mean.copy_(new_mean)
        self.mean_square.copy_(new_mean_square)
        self.std.copy_(new_std)
        self.updates += 1
        return self.norm(targets).view(*targets_shape)

    def norm(self, x):
        return (x - self.mean) / self.std

    def unnorm(self, value):
        return value * self.std + self.mean


class QNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, 1)

        self.ln1 = LayerNorm(hidden_dim)
        self.ln2 = LayerNorm(hidden_dim)
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.ln1(self.linear1(xu)))
        x1 = F.relu(self.ln2(self.linear2(x1)))
        x1 = self.last_fc(x1)

        return x1


class Beta_network(nn.Module):

    def __init__(self, num_inputs, num_options, hidden_dim):
        super().__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_options)

        self.ln1 = LayerNorm(hidden_dim)
        self.ln2 = LayerNorm(hidden_dim)

        self.apply(weights_init_)

    def forward(self, input):
        x1 = F.relu(self.ln1(self.linear1(input)))
        x1 = F.relu(self.ln2(self.linear2(x1)))
        x1 = torch.sigmoid(self.last_fc(x1))

        return x1


class Q_discrete_Network(nn.Module):

    def __init__(self, num_inputs, num_options, hidden_dim):
        super(Q_discrete_Network, self).__init__()
        self.num_options = num_options
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_options)

        self.ln1 = LayerNorm(hidden_dim)
        self.ln2 = LayerNorm(hidden_dim)

        self.apply(weights_init_)

    def forward(self, state):
        h = F.relu(self.ln1(self.linear1(state)))
        h = F.relu(self.ln2(self.linear2(h)))
        x1 = self.last_fc(h)

        return x1

    def sample_option(self, state):
        x = self.forward(state)
        return torch.softmax(x, dim=-1)


class TwinnedQNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
        self.Q1 = QNetwork(num_inputs, num_actions, hidden_dim)
        self.Q2 = QNetwork(num_inputs, num_actions, hidden_dim)

    def forward(self, states, action):
        q1 = self.Q1(states, action)
        q2 = self.Q2(states, action)
        return q1, q2


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
            pre_tanh_value = (torch.log((1 + value) / (1 - value)) / 2).clamp(-1 + 1e-10, 1 - 1e-10)

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


class GaussianPolicy(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.ln1 = LayerNorm(hidden_dim)
        self.ln2 = LayerNorm(hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.ln1(self.linear1(state)))
        x = F.relu(self.ln2(self.linear2(x)))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(
            self,
            state,
            deterministic=False,
            return_log_prob=False,
    ):

        mean, log_std = self.forward(state)
        std = log_std.exp()
        tanh_normal = TanhNormal(mean, std)

        log_prob = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            if return_log_prob:
                action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                action = tanh_normal.rsample()

        return action, log_prob

    def get_logp(self, state, action, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        tanh_normal = TanhNormal(mean, std)

        return tanh_normal.log_prob(action)
