from smac.env import StarCraft2Env
from envs.starcraft2.messy_sc2 import NoisyStarCraft
import numpy as np
import sys
from datetime import datetime
import time
import os
from os.path import join

map_name = sys.argv[1]
n_episodes = int(sys.argv[2])
max_steps = 25

time.sleep(np.random.randint(0, 60))
datetimetext = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = "nstates_{}_{}.npy".format(map_name, datetimetext)
#obs_filename = "nobservations_{}_{}.npy".format(map_name, datetimetext)
filename = "nstates_{}_{}.npy".format(map_name, datetimetext)
obs_filename = "nobservations_{}_{}.npy".format(map_name, datetimetext)
dirname = "../init_states"
if not os.path.exists(dirname):
    os.makedirs(dirname)
save_path = join(dirname, filename)
obs_save_path = join(dirname, obs_filename)

def main(map_name, n_episodes):
    recorded_states = []
    recorded_observations = []
    #env = StarCraft2Env(map_name=map_name)
    env = NoisyStarCraft(map_name=map_name)
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    total_states = 0
    for e in range(n_episodes):
        env.reset()
        terminated = False
        steps = 0
        states = []
        observations = []
        print("Running episode", e)
        while not terminated:# and steps <= max_steps:
            state = env.get_state()
            obs = env.get_obs()
            state = np.array(state)
            states.append(state)
            obs = np.array(obs)
            observations.append(obs)
            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)
            _, terminated, _ = env.step(actions)
            steps += 1
        total_states += steps
        recorded_states.append(states)
        recorded_observations.append(observations)
    env.close()
    np.save(save_path, recorded_states)
    np.save(obs_save_path, recorded_observations)
    print(total_states, "steps saved")

main(map_name, n_episodes)