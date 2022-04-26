import tensorflow as tf
import keras
from keras import layers

action = tf.random.categorical([[0.3, 0.3, 0.3]], 1)
print(action)
print(tf.__version__)
print(tf.keras)
