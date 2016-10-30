import tensorflow as tf
import numpy as np

# placeholder vector 2 rows with datatype int32
a = tf.placeholder(tf.int32, shape=[2], name="input_a")

#Use the placeholder
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")

#Finish the graph
d = tf.add(b, c, name="add_d")

sess = tf.Session()

input_dict = {a: np.array([5,3], dtype=np.int32)}

# Fetch the value of `d`, therefore feeding the values of `input_vector` into `a`
print(sess.run(d, feed_dict=input_dict))
# operation d which is addition is being fed input_dict via feed_dict [5,3]
