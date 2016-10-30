import tensorflow as tf
import numpy as np

"""
Generating new graphs
"""

g1 = tf.Graph()
g2 = tf.Graph()

with g1.as_default():
    a = tf.add(2,3, name="add_a")
    b = tf.mul(3,3, name="mul_b")

with g2.as_default():
    c = tf.div(4, 2, name="div_c")

"""
Assign a handle to default graphs
"""

g3 = tf.get_default_graph()
g4 = tf.Graph()

with g3.as_default():
    # Do something here

with g4.as_default():
    #Do something here
