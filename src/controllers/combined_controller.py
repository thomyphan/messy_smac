from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from controllers.basic_controller import BasicMAC
import torch as th
import random

# This multi-agent controller shares parameters between agents and replaces some by an alternative policy
class CombinedMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(CombinedMAC, self).__init__(scheme, groups, args)
        self.n_replacements = args.n_replacements
        assert self.n_replacements <= self.n_agents
        self.replacement_ids = random.sample(range(self.n_agents), k=self.n_replacements)
        self.original_ids = [i for i in range(self.n_agents) if i not in self.replacement_ids]
        self.n_original = self.n_agents - self.n_replacements

    def forward(self, ep_batch, t, test_mode=False):
        batch_size_orignal = ep_batch.batch_size * self.n_original
        batch_size_replacements = ep_batch.batch_size * self.n_replacements
        agent_inputs = self._build_inputs(ep_batch, t).view(ep_batch.batch_size, self.n_agents, -1)
        avail_actions = ep_batch["avail_actions"][:, t]
        out_dtype = self.original_hidden_states.dtype
        out_device = self.original_hidden_states.device
        out_layout = self.original_hidden_states.layout
        agent_outs = th.zeros([ep_batch.batch_size, self.n_agents, self.args.n_actions], dtype=out_dtype, device=out_device, layout=out_layout)
        original_outs, self.original_hidden_states = self.agent(agent_inputs[:, self.original_ids, :]\
            .view(batch_size_orignal, -1), self.original_hidden_states)
        replacement_outs, self.replacement_hidden_states = self.replacement_agent(agent_inputs[:, self.replacement_ids, :]\
            .view(batch_size_replacements, -1), self.replacement_hidden_states)
        self.hidden_states[:, self.original_ids, :] = self.original_hidden_states.view(ep_batch.batch_size, self.n_original, -1)
        self.hidden_states[:, self.replacement_ids, :] = self.replacement_hidden_states.view(ep_batch.batch_size, self.n_replacements, -1)
        assert self.hidden_states[:, self.original_ids, :].requires_grad
        assert self.hidden_states[:, self.replacement_ids, :].requires_grad
        agent_outs[:, self.original_ids, :] = original_outs.view(ep_batch.batch_size, self.n_original, -1)
        agent_outs[:, self.replacement_ids, :] = replacement_outs.view(ep_batch.batch_size, self.n_replacements, -1)
        assert agent_outs[:, self.original_ids, :].requires_grad
        assert agent_outs[:, self.replacement_ids, :].requires_grad

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.original_hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_original, -1)  # bav
        self.replacement_hidden_states = self.replacement_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_replacements, -1)  # bav
        out_dtype = self.original_hidden_states.dtype
        out_device = self.original_hidden_states.device
        out_layout = self.original_hidden_states.layout
        self.hidden_states = th.zeros([batch_size, self.n_agents, self.original_hidden_states.size(2)], dtype=out_dtype, device=out_device, layout=out_layout)
        self.hidden_states[:, self.original_ids, :] = self.original_hidden_states
        self.hidden_states[:, self.replacement_ids, :] = self.replacement_hidden_states

    def parameters(self, is_replacement=False):
        if not is_replacement:
            return self.agent.parameters()
        return self.replacement_agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.replacement_agent.load_state_dict(other_mac.replacement_agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.replacement_agent.cude()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.agent.state_dict(), "{}/replacement_agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.replacement_agent.load_state_dict(th.load("{}/replacement_agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.replacement_agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
