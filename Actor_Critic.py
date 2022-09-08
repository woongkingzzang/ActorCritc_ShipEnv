'''
def뒤에 -> return 값의 형식을 표시하기 위함

한 방향으로만 회전, 그에 따라 reward가 쌓이지 않음

1. render 코드의 위치의 문제?
    딱히 문제 되지 않는 듯
    
2. action 정의가 문제??
    action이 2만 들어감 => action 정의를 바꿔주니 직진만 함 = action 정의 문제인듯함
    
'''

from argparse import Action
import collections
from dis import dis
from time import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
import gym
# import keras

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

from gym.envs.registration import register
from gym_env import ShipEnv

register(
    id='ShipEnv-v0',
    entry_point='gym_env:ShipEnv',
)

env = gym.make("ShipEnv-v0")

seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

eps = np.finfo(np.float32).eps.item()

class ActorCritic(tf.keras.Model):
    '''Combined actor-critic network'''
    
    def __init__(
        self,
        num_actions:int,
        num_hidden_units: int):
        
        super().__init__()
        self.f1 = layers.Dense(num_hidden_units, activation= 'relu')
        self.f2 = layers.Dense(num_hidden_units/2, activation= 'relu')
        self.f3 = layers.Dense(num_hidden_units/4, activation= 'relu')
        self.f4 = layers.Dense(num_hidden_units/8, activation= 'relu')
        self.f5 = layers.Dense(num_hidden_units/16, activation = "relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.f1(inputs)
        x1 = self.f2(x)
        x2 = self.f3(x1)
        x3 = self.f4(x2)
        x4 = self.f5(x3)
        return self.actor(x4), self.critic(x4)



num_actions = env.action_space.n # env.action_space.n
# num_actions = 3
num_hidden_units = 1024
model = ActorCritic(num_actions, num_hidden_units)


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, reward, done,_ = env.step(action)

    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    
    return tf.numpy_function(env_step, [action], 
                           [tf.float32, tf.int32, tf.int32])

def run_episode(
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int) -> List[tf.Tensor]:

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state
    for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(state)
        # action_logits_t = [[0.,1.,2.]]

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0] # sample_number = 3
        # print(tf.random.categorical(action_logits_t, 3)[:])
        action_probs_t = tf.nn.softmax(action_logits_t)


        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)

        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break
  
    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()
    
  
    return action_probs, values, rewards

def get_expected_return(
    rewards: tf.Tensor, 
    gamma: float, 
    standardize: bool = True) -> tf.Tensor:
    

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / 
                    (tf.math.reduce_std(returns) + eps))

    return returns

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
    action_probs: tf.Tensor,  
    values: tf.Tensor,  
    returns: tf.Tensor) -> tf.Tensor:

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


@tf.function
def train_step(
    initial_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> tf.Tensor:
 

    with tf.GradientTape() as tape:
        
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        action_probs, values, rewards = run_episode(
            initial_state, model, max_steps_per_episode) 

        returns = get_expected_return(rewards, gamma)

        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

        loss = compute_loss(action_probs, values, returns)

    # Compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    
    return episode_reward


max_episodes = 10000
max_steps_per_episode = 1000

reward_threshold = 1000
running_reward = 0

gamma = 0.99
with tqdm.trange(max_episodes) as t:
  for i in t:

    initial_state = tf.constant(env.reset(), dtype=tf.float32)
    # env.render()
    episode_reward = int(train_step(
        initial_state, model, optimizer, gamma, max_steps_per_episode))
    
    running_reward = episode_reward*0.01 + running_reward*0.99
    t.set_description(f'Episode {i}')
    t.set_postfix(
        episode_reward=episode_reward, running_reward=running_reward)

    if i % 10 == 0:
      pass # print(f'Episode {i}: average reward: {avg_reward}')
  
    if running_reward > reward_threshold:  
        break
    
print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


### Visualization
from PIL import Image
from pyvirtualdisplay import Display

display = Display(visible = 0, size = (800,800))
display.start()

def render_episode(env: gym.Env, model: tf.keras.Model, max_steps: int): 
  screen = env.render(mode='rgb_array')
  im = Image.fromarray(screen)

  images = [im]

  state = tf.constant(env.reset(), dtype=tf.float32)
  for i in range(1, max_steps + 1):
    state = tf.expand_dims(state, 0)
    action_probs, _ = model(state)
    action = np.argmax(np.squeeze(action_probs))

    state, _, done, _ = env.step(action)
    state = tf.constant(state, dtype=tf.float32)

    # Render screen every 10 steps
    if i % 10 == 0:
      screen = env.render(mode='rgb_array')
      images.append(Image.fromarray(screen))

    if done:
      break

  return images  

# Save GIF image
images = render_episode(env, model, max_steps_per_episode)
image_file_0 = 'shipenv-v0.gif'
image_file_1 = 'shipenv-v1.gif'
# loop=0: loop forever, duration=1: play each frame for 1ms
images[0].save(
    image_file_0, save_all=True, append_images=images[1:], loop=0, duration=1)

images[len(images)-1].save(
    image_file_1, save_all=True, append_images=images[1:], loop=0, duration=1
    
# import tensorflow_docs.vis.embed as embed
# embed.embed_file(image_file)