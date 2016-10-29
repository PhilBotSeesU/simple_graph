import tensorflow as tf
import numpy as np

#0-dim tensor with 32 bit datatype
t0 = np.array(50, dtype=np.int32)

#1-dimension tensor with byte string datatype, aka a vector
t1 = np.array([b"apply", b"peaches", b"grapes"])

#2-dimension tensor with boolean datatypes
t2 = np.array([[True, False, True],
              [True, True, False],
              [False, False, True]], dtype=np.bool)

#3-dimension tensor with 64-bit datatype
t3 = np.array([[0,0], [0,1], [0,2],
               [1,2], [1,1], [1,1]], dtype=np.int64)

#2-dimension tensor with 32-bit datatype
t4 = np.array([[11,21], [3,3]], dtype=np.int32)

"""Shapes"""
single_list = []
single_tuple = ()

#Shape that describes a vector of 3
s1 = [3]

#3 by 2 matrix
#e.g
"""
[
[1,2],
[2,2],
[4,5]
]
"""
s2 = (3,2)

#Take any shape
s1_flex = [None]

#Shape of matrix any rows long but with 3 rows
s2_flex = (None, 3)

# Shape of 3-D tensor with length (aka rows) 2 in first dimension
# and variable lengths in next 2 dimensions
s3_flex = [2, None, None]
shape = tf.shape(t3, name="mystery_shape")
sess = tf.Session()
print(sess.run(shape))


#Any type of tensor!
s2_any = None
