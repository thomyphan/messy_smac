from smac.env import StarCraft2Env
from utils.dict2namedtuple import convert
import numpy as np

class MessyStarCraft(StarCraft2Env):
    def __init__(self, **kwargs):
        # Unpack arguments from sacred
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.init_random_steps = getattr(args, "randomize_initial_state", 10)
        self.failure_obs_prob = getattr(args, "failure_obs_prob", 0.15)
        args = args._asdict()
        args.pop("randomize_initial_state", 10)
        args.pop("failure_obs_prob", 0.15)
        super(MessyStarCraft, self).__init__(**args)

    def get_obs_agent(self, agent_id):
        default_obs = super(MessyStarCraft, self).get_obs_agent(agent_id)
        if np.random.rand() <= self.failure_obs_prob:
            return -1.0*default_obs
        return default_obs

    def reset(self):
        super(MessyStarCraft, self).reset()
        terminated = False
        steps = 0
        nr_random_steps = self.init_random_steps
        while steps < nr_random_steps:
            actions = []
            for agent_id in range(self.n_agents):
                avail_actions = self.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)
            _, terminated, _ = self.step(actions)
            steps += 1
            if terminated:
                super(MessyStarCraft, self).reset()
        observations = self.get_obs()
        return [np.zeros_like(obs) for obs in observations], self.get_state()

