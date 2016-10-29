import tensorflow as tf

"""Define graph"""
a = tf.constant([5,3], name="input_a")
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")
d = tf.add(b,c, name="add_d")

"""Execute graph"""
sess = tf.Session()

"""Output"""
print(sess.run(b))
print(sess.run(c))
print(sess.run(d))

sess.close()
