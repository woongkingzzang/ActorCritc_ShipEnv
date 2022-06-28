import tensorflow as tf
from tensorflow import keras

action = tf.random.categorical([[0.3, 0.3, 0.3]], 1)
print(action)
print(tf.__version__)
print(keras)
