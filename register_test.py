from email import policy
from gym.envs.registration import register
import gym
from gym_env_case5 import ShipEnv

register(
    id='ShipEnv-v0',
    entry_point='gym_env:ShipEnv',
)

env = gym.make('ShipEnv-v0')
# env.reset()
observation,info = env.reset(seed=42, return_info=True)
rewards = 0
for _ in range(2000):
    env.render()
    done = False
    action = env.action_space.sample()

    # state, reward, done, _ = env.step(env.action_space())
    observation, reward, done, info = env.step(action)
    rewards += reward
    print("rewards", rewards)
    if done:
        observation, info = env.reset(return_info=True)
        rewards = 0
        # print("##############True###############")
    
    
env.close()