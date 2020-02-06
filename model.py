import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        lin1 = nn.Linear(input_dim, 128)
        torch.nn.init.xavier_normal_(lin1.weight)
        lin2 = nn.Linear(128, 64)
        torch.nn.init.xavier_normal_(lin2.weight)
        lin3 = nn.Linear(64, output_dim)
        torch.nn.init.xavier_normal_(lin3.weight)
        self.layers = nn.Sequential(lin1, nn.ReLU(), lin2, nn.ReLU(), lin3)

    def forward(self, input):
        return self.layers(input)


class Critic(Model):

    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim + action_dim, 1)

    def forward(self, x, a):
        return super().forward(torch.cat((x, a), 1))


class Actor(Model):

    def __init__(self, state_dim):
        super().__init__(state_dim, 1)

    def forward(self, x):
        return torch.tanh(super().forward(x))