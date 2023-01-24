import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.centralV import CentralVCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop
import numpy as np

class MAPPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger
        self.mac_input_shape = int(np.prod(mac._get_input_shape(scheme)))
        self.history_dim = [(self.args.episode_limit+1), args.n_agents, self.mac_input_shape]
        self.ppo_epoch = args.ppo_epoch
        self.ppo_clip_param = args.ppo_clip_param
        self.state_history = args.state_history
        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.individual_input_shape = mac._get_input_shape(scheme)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.critic = CentralVCritic(scheme, self.individual_input_shape, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        critic_mask = mask.clone()
        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        old_log_pi_taken = None
        for _ in range(self.ppo_epoch):

            actions = batch["actions"][:, :][:,:-1]

            observations = []
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.mac.forward(batch, t=t) # Shape [batch_size, n_agents, n_actions]
                mac_out.append(agent_outs)
                inputs = self.mac._build_inputs(batch, t=t).view(batch.batch_size, self.args.n_agents, self.mac_input_shape)
                observations.append(inputs)
            mac_out = th.stack(mac_out[:-1], dim=1)  # Concat over time
            observations = th.stack(observations, dim=1)
            # Shape [batch_size, max_len, nr_agents, 1]
            states = batch["state"]
            target_states = batch["state"]
            advantages = self._train_critic(states, observations, target_states, rewards, terminated, critic_mask)

            # Mask out unavailable actions, renormalise (as in action selection)
            mac_out[avail_actions == 0] = 0
            mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
            mac_out[avail_actions == 0] = 0
            pi = mac_out.view(-1, self.n_actions)
            # Calculate policy grad with mask
            pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
            pi_taken[mask == 0] = 1.0
            log_pi_taken = th.log(pi_taken)
            if old_log_pi_taken is None:
                old_log_pi_taken = log_pi_taken.clone().detach()

            imp_weights = th.exp(log_pi_taken - old_log_pi_taken)

            surr1 = imp_weights * advantages
            surr2 = th.clamp(imp_weights, 1.0 - self.ppo_clip_param, 1.0 + self.ppo_clip_param) * advantages
            mappo_loss = (-th.sum(th.min(surr1, surr2),dim=-1, keepdim=True) * mask).sum() / mask.sum()

            # Optimise agents
            self.agent_optimiser.zero_grad()
            mappo_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

            if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
                self._update_targets()
                self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def _train_critic(self, states, observations, target_states, rewards, terminated, mask):
        # Optimise critic
        targets_taken = self.target_critic(target_states, observations).squeeze(3)
        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda)
        targets_mean = targets.mean()
        targets_std = targets.std()
        targets = (targets - targets_mean) / (targets_std + np.finfo(float).eps)
        q_vals = self.critic(states, observations).squeeze(3)
        td_error = (q_vals[:,:-1] - targets.detach())
        mask = mask.expand_as(td_error)
        assert td_error.size() == mask.size(), "Expected {}, got {}".format(mask.size(), td_error.size())
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.critic_optimiser.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.critic_training_steps += rewards.size(1)
        advantages = targets - q_vals[:,:-1]
        return advantages.detach().reshape(-1)

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        pass

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
