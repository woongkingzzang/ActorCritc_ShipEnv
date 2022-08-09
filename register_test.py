from email import policy
from gym.envs.registration import register
import gym
from gym_env import ShipEnv

register(
    id='ShipEnv-v0',
    entry_point='gym_env:ShipEnv',
)

env = gym.make('ShipEnv-v0')
# env.reset()
observation,info = env.reset(seed=42, return_info=True)
for _ in range(2000):
    env.render()
    done = False
    action = env.action_space.sample()

    # state, reward, done, _ = env.step(env.action_space())
    observation, reward, done, info = env.step(action)
    print("reward", reward)
    print("step",_)
    print("done", done)
    if done:
        observation, info = env.reset(return_info=True)
        print("##############True###############")

    
env.close()
