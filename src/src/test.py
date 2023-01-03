import envs.dec_tiger.dec_tiger as dec_tiger
import numpy

args = {}
env = dec_tiger.DecTiger(batch_size=None, env_args=args)
env.reset()
state = env.get_state()
joint_action = numpy.array([2,0])
reward, done, _ = env.step(joint_action)
obs = env.get_obs()

print("state:", state)
print("actions:", joint_action)
print("reward:", reward)
print("done:", done)
print("state:", obs)