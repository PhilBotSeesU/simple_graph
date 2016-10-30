import tensorflow as tf
import numpy as np

"""
A number of session exercises
"""

a = tf.add(2,5)
b = tf.mul(a, 3)

#open session, specify a new graph or leave empty for default. i.e. showing how to specify a graph
#Takes operations - fetches (data)
# if tensor, output numpy array. Operation then None
sess = tf.Session(graph=tf.get_default_graph())

#computes but returns nothing from all variables


"""
feed_dict : Used to override all the values in the graph
Expects python dictionary as input
Keys in dictionary are handles to Tensor Objects that should be overridden
values = strings, numbers, arrays, etc
Values MUST BE ALL OF THE SAME TYPE
Or converted to the same type i.e. int32 to int64 not string to int32
tensor key data type == value data type

"""

replace_dict = {a: 15}

sess.run(b, feed_dict=replace_dict)

sess.close()
