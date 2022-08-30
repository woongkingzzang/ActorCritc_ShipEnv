import tensorflow as tf
from tensorflow import keras
a = tf.tensor.stack
for _ in range(1000):
    action = tf.random.categorical([[0.3, 0.2, 0.1]], 1)[0,0]
    print(action)

