from email import policy
from gym.envs.registration import register
import gym
from gym_env import ShipEnv
import pandas as pd
import os

register(
    id='ShipEnv-v0',
    entry_point='gym_env:ShipEnv',
)

env = gym.make('ShipEnv-v0')
# env.reset()
observation,info = env.reset(seed=42, return_info=True)
rewards = 0
x_list = []
y_list = []
ts_x_list = []
ts_y_list = []
# for _ in range(2000):
#     env.render()
#     done = False
action_list = [2, 0, 2, 2, 2, 2, 3, 4, 0, 2, 3, 4, 1, 4, 4, 3, 2, 1, 3, 2, 0, 1, 4, 2, 0, 3, 4, 2, 4, 4, 1, 4, 2, 1, 1, 0, 1, 2, 3, 1, 2, 3, 3, 0, 0, 0, 0, 2, 0, 0, 0, 3, 4, 0, 2, 4, 3, 4, 4, 2, 1, 2, 1, 2, 3, 2, 3, 4, 0, 3, 3, 2, 0, 0, 3, 0, 1, 2, 2, 2, 0, 1, 2, 1, 3, 1, 2, 4, 2, 2, 2, 2, 3, 2, 0, 2, 4, 3, 2, 2, 2, 4, 3, 2, 2, 4, 4, 4, 0, 0, 3, 0, 2, 0, 2, 0, 3, 3, 3, 2, 0, 1, 4, 0, 2, 2, 2, 1, 1, 4, 0, 1, 0, 2, 4, 2, 3, 1, 4, 1, 1, 3, 4, 4, 4, 0, 2, 0, 0, 2, 3, 0, 2, 4, 2, 2, 0, 1, 2, 0, 1, 4, 2, 1, 4, 2, 4, 2, 4, 3, 0, 2, 2, 1, 0, 4, 3, 0, 3, 2, 1, 1, 2, 0, 3, 0, 1, 4, 2, 3, 0, 2, 0, 2, 0, 2, 2, 0, 0, 0, 2, 4, 3, 0, 0, 0, 4, 3, 2, 3, 2, 1, 0, 2, 2, 4, 4, 0, 2, 2, 2, 3, 3, 3, 2, 3, 0, 1, 2, 2, 2, 0, 2, 2, 4, 3, 1, 0, 0, 1, 4, 3, 4, 1, 3, 0, 4, 0, 0, 2, 3, 2, 1, 3, 0, 3, 1, 3, 0, 0, 1, 3, 1, 2, 1, 2, 4, 1, 2, 1, 2, 4, 2, 2, 0, 0, 3, 1, 4, 1, 0, 1, 2, 3, 2, 1, 3, 2, 4, 4, 0, 3, 3, 1, 4, 0, 4, 4, 4, 0, 2, 3, 1, 4, 0, 3, 4, 2, 4, 2, 0, 4, 3, 2, 4, 2, 3, 4, 0, 0, 2, 2, 3, 2, 2, 2, 2, 0, 0, 0, 3, 2, 3, 0, 2, 2, 3, 4, 2, 4, 4, 2, 3, 2, 0, 0, 0, 2, 0, 2, 3, 3, 4, 1, 0, 3, 3, 4, 0, 2, 0, 4, 3, 0, 0, 4, 3, 4, 4, 4, 3, 0, 1, 4, 1, 0, 4, 0, 0, 4, 4, 0, 2, 3, 4, 0, 1, 2, 0, 1, 4, 2, 1, 1, 2, 0, 1, 4, 0, 0, 1, 3, 2, 0, 2, 2, 2, 1, 2, 2, 0, 3, 2, 0, 4, 1, 3, 1, 4, 4, 2, 3, 2, 1, 2, 4, 2, 1, 0, 4, 2, 4, 1, 0, 0, 3, 4, 1, 2, 1, 2, 0, 2, 4, 1, 0, 2, 1, 3, 1, 1, 0, 1, 0, 0, 3, 1, 2, 0, 3, 4, 4, 0, 2, 1, 1, 1, 0, 2, 4, 1, 4, 0, 1, 0, 0, 1, 4, 2, 3, 1, 1, 3, 1, 0, 4, 4, 2, 3, 1, 2, 2, 4, 4, 0, 4, 3, 2, 4, 3, 1, 0, 2, 2, 3, 4, 2, 2, 1, 2, 3, 0, 2, 3, 0, 0, 4, 3, 4, 0, 2, 2, 0, 0, 3, 3, 4, 0, 2, 1, 2, 1, 1, 1, 4, 4, 2, 0, 2, 2, 0, 0, 3, 2, 4, 1, 1, 2, 0, 2, 0, 0, 4, 3, 4, 1, 0, 0, 0, 2, 1, 0, 4, 0, 4, 4, 4, 2, 3, 0, 2, 4, 0, 2, 2, 4, 1, 2, 2, 4, 0, 1, 2, 2, 2, 1, 4, 3, 3, 1, 0, 1, 1, 4, 1, 0, 1, 1, 2, 2, 4, 3, 0, 1, 2, 2, 1, 3, 0, 0, 2, 0, 2, 2, 0]

for i in range(len(action_list)):
    env.render()
    done = False
    action = action_list[i]
    observation, reward, done, info = env.step(action)
    rewards += reward
    x_list.append(env.position_x)
    y_list.append(env.position_y)
    ts_x_list.append(env.ts_pos_x)
    ts_y_list.append(env.ts_pos_y)
    print("rewards", rewards)

    if done:
        observation, info = env.reset(return_info=True)
        rewards = 0
    if i == len(action_list):
        break

xy_df = pd.DataFrame({'os_x': x_list, 'os_y':y_list, 'ts_x':ts_x_list, 'ts_y': ts_y_list})
xy_df.to_csv('/home/phl1/daewoong_ws/result.csv')

    
env.close()
