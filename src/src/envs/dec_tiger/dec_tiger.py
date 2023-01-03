from numpy.core.numeric import indices
from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np

int_type = np.int16
float_type = np.float32

OPEN_LEFT = 0
OPEN_RIGHT = 1
LISTEN = 2

class DecTiger(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        # Unpack arguments from sacred
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)
        self.truncate_episodes = getattr(args, "truncate_episodes", True)
        self.fully_observable = getattr(args, "fully_observable", False)
        if self.fully_observable:
            self.failure_obs_prob = 0
        else:
            self.failure_obs_prob = getattr(args, "failure_obs_prob", 0.15)

        # Define the agents and actions
        self.n_agents = 2
        self.n_actions = 3 # Listen Left/Listen Right/Nothing
        self.episode_limit = 4

        self.batch_mode = batch_size is not None
        self.batch_size = batch_size if self.batch_mode else 1
        self.terminated = [False for _ in range(self.batch_size)]
        self.all_obs = np.zeros((self.batch_size, self.n_agents, 2))
        self.state = np.zeros((self.batch_size, 2))
        self.steps = 0

    def reset(self):
        """ Returns initial observations and states"""
        self.terminated = [False for _ in range(self.batch_size)]
        self.steps = 0
        self.state[:] = 0
        self.all_obs[:] = 0
        tiger_positions = np.rint(np.random.rand(self.batch_size))
        tiger_observations = np.rint(np.random.rand(self.batch_size, self.n_agents))
        for b in range(self.batch_size):
            non_zero = tiger_positions[b] > 0
            state_index = int(tiger_positions[b])
            self.state[b][state_index] = 1
            assert not non_zero or state_index == 1
            for a in range(self.n_agents):
                non_zero = tiger_observations[b][a] > 0
                obs_index = int(tiger_observations[b][a])
                assert not non_zero or obs_index == 1
                if self.fully_observable:
                    obs_index = state_index
                self.all_obs[b][a][obs_index] = 1
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Execute a*bs actions in the environment. """
        if not self.batch_mode:
            actions = np.expand_dims(np.asarray(actions, dtype=int_type), axis=1)
        assert len(actions.shape) == 2 and actions.shape[0] == self.n_agents and actions.shape[1] == self.batch_size, \
            "improper number of agents and/or parallel environments!"
        actions = actions.astype(dtype=int_type)
        reward = np.zeros(self.batch_size, dtype=float_type)
        self.all_obs[:] = 0
        for b in [i for i in range(self.batch_size) if not self.terminated[i]]:
            action_0 = actions[0, b]
            action_1 = actions[1, b]
            if action_0 == action_1:
                if action_0 == LISTEN:
                    reward[b] = -2
                    for a in range(self.n_agents):
                        rand = np.random.rand()
                        tiger_position = np.argmax(self.state[b])
                        assert tiger_position in [0,1], "unknown tiger position '{}'".format(tiger_position)
                        if rand <= self.failure_obs_prob:
                            other_position = (tiger_position+1)%2
                            self.all_obs[b][a][other_position] = 1
                        else:
                            self.all_obs[b][a][tiger_position] = 1
                else:
                    self.terminated[b] = True
                    if self.state[b][action_0] < 1:
                        reward[b] = 20
                    else:
                        reward[b] = -50
                    for a in range(self.n_agents):
                        random_index = int(np.rint(np.random.rand()))
                        self.all_obs[b][a][random_index] = 1
            else:
                rew = []
                for a in range(self.n_agents):
                    action = actions[a,b]
                    if action == LISTEN:
                        reward[b] = -1
                    else:
                        self.terminated[b] = True
                        if self.state[b][action] < 1:
                            rew.append(10)
                        else:
                            rew.append(-100)
                    random_index = int(np.rint(np.random.rand()))
                    self.all_obs[b][a][random_index] = 1
                if len(rew) > 0:
                    reward[b] += min(rew)
                assert reward[b] <= -100 or int(reward[b]) == 9, "unknown outcome with reward '{}'".format(reward[b])  
        self.steps += 1
        info = {}
        if self.steps >= self.episode_limit:
            self.terminated = [True for _ in range(self.batch_size)]
            info["episode_limit"] = self.truncate_episodes
        else:
            info["episode_limit"] = False
        if self.batch_mode:
            return reward, self.terminated, info
        else:
            return reward[0].item(), int(self.terminated[0]), info

    def get_obs(self):
        return self.all_obs

    def get_obs_agent(self, agent_id, batch=0):
        """ Returns observation for agent_id """
        return self.all_obs[batch][agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.all_obs.shape[2]

    def get_state(self):
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.state.shape

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.n_actions))

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_stats(self):
        pass

