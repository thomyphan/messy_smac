import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop

class RepresentationRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RepresentationRNNAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.decoder = nn.Linear(args.rnn_hidden_dim, input_shape)
        self.params = list(self.parameters())
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward_all(self, inputs, hidden_state, detach_hidden=True):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        if detach_hidden:
            h = h.detach()
        dec = self.decoder(h)
        q = self.fc2(h)
        q = F.relu(q)
        q = self.fc3(q)
        return q, h, dec

    def forward(self, inputs, hidden_state):
        q, h, _ = self.forward_all(inputs, hidden_state, detach_hidden=True)
        return q, h

    def train_representation(self, inputs, batch_size):
        hidden_states = self.init_hidden().unsqueeze(0).expand(batch_size, self.args.n_agents, -1)
        current_inputs = inputs[:-1]
        next_inputs = inputs[1:]
        losses = []
        for current, next in zip(current_inputs, next_inputs):
            _, hidden_states, dec = self.forward_all(current, hidden_states, detach_hidden=False)
            losses.append(F.mse_loss(next, dec))
        loss = th.stack(losses).sum()
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

    
        
