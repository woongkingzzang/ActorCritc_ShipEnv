from gym.envs.registration import register
import gym
from gym_env import ShipEnv

register(
    id='ShipEnv-v0',
    entry_point='gym_env:ShipEnv',
)

env = gym.make('ShipEnv-v0')
env.reset()

for _ in range(2000):
    env.render()
    env.step(env.action_space.sample())
    # state, reward, done, _ = env.step(env.action_space())
    # if done:
        # env.reset()
    
env.close()