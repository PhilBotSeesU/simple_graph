import tensorflow as tf
import numpy as np

"""Initialize some tensors"""
a = np.array([2,3], dtype=np.int32)
b = np.array([4,5], dtype=np.int32)

#Use tf.add() for initializing the operation
c = tf.add(a,b)
