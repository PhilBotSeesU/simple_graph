import tensorflow as tf

"""#Step 1 - Define the graph"""
a = tf.constant(5, name="in_a")
b = tf.constant(3, name="in_b") # takes single tensor value, outputs same value

c = tf.mul(a,b, name="mul_c") #takes two tensors and multiplies them as an output product
d = tf.add(a,b, name="add_d")
e = tf.add(c,d, name="add_e")

""" Step 2 - execute the graph """
