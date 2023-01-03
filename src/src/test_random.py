from smac.env import StarCraft2Env
import numpy as np
import sys

map_name = sys.argv[1]
n_episodes = int(sys.argv[2])
max_steps = 0
if len(sys.argv) > 3:
    max_steps = int(sys.argv[3])

def main(map_name, n_episodes):
    env = StarCraft2Env(map_name=map_name)
    env_info = env.get_env_info()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    states = []
    for e in range(n_episodes):
        env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < max_steps:
            state = env.get_state()
            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)
            _, terminated, _ = env.step(actions)
            steps += 1
        state = np.array(env.get_state())
        states.append(state.reshape(-1))
        print("loading episode", e, "->", steps, "steps")
    env.close()
    max_value = np.max(states, axis=0)
    min_value = np.min(states, axis=0)
    value_range = (max_value - min_value)*1.0
    mean = np.mean(states, axis=0)
    assert len(mean) == len(states[0])
    std_max = max(np.std(states, axis=0))
    dist_max = max([sum(abs(state - mean)) for state in states])
    print("max_std:      ",std_max)
    print("max_dist:     ",dist_max)
    print("value_range:  ",max(value_range))

main(map_name, n_episodes)