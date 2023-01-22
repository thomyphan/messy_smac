import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CentralVCritic(nn.Module):
    def __init__(self, scheme, individual_input_shape, args):
        super(CentralVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.state_dim = int(np.prod(args.state_shape))
        self.observation_dim = int(np.prod(individual_input_shape))

        # Set up network layers
        self.fc1 = nn.Linear(self.state_dim + self.observation_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, states, observations, t=None):
        bs = states.size(0)
        states = states.view(bs, -1, 1, self.state_dim).repeat(1, 1, self.n_agents, 1)
        states = states.reshape(-1, self.state_dim)
        observations = observations.reshape(-1, self.observation_dim)
        inputs = th.concat([states, observations], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q.view(bs, -1, self.n_agents, 1)