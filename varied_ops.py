import tensorflow as tf

"""Define the graph"""
a = tf.constant(10, name="input_a")
b = tf.constant(5, name="input_b")
x = tf.constant(20, name="input_x")
y = tf.constant(40, name="input_y")

c = tf.sub(a, b, name="sub_c")
d = tf.div(a, b, name="div_d" )

f = tf.div(x, c, name="div_f")

e = tf.mod(c, d, name="mod_e")

""" Execute the graph variables"""
sess = tf.Session() # Create session management object
output = sess.run(c)

print "c variable ", output #Can print this way, too
print("d:", sess.run(d))
print("e:", sess.run(e))
print("f: ", sess.run(f))

writer = tf.train.SummaryWriter('./my_graph', sess.graph) #Track graph ops executed

writer.close() #Close
sess.close() #Close
