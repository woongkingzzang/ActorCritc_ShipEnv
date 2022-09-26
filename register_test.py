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
rewards = 0
for _ in range(4000):
    env.render()
    done = False
    action = env.action_space.sample()

    # state, reward, done, _ = env.step(env.action_space())
    observation, reward, done, info = env.step(action)
    rewards += reward
    print("action", action)
    print("rewards", rewards)
    # if rewards < -1000:
    #     done == True
    if done:
        observation, info = env.reset(return_info=True)
        rewards = 0
        # print("##############True###############")
    

    
env.close()
