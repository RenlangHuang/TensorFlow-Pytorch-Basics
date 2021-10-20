import numpy as np
import tensorflow as tf

a = np.arange(0, 5)
b = tf.convert_to_tensor(a, dtype=tf.int64) # default
print("numpy:", a)
print("tensorflow:", b)
print('compare:',tf.convert_to_tensor(np.arange(0, 5),dtype=tf.int32))

a = np.random.random((3,4))
b = tf.convert_to_tensor(a)
print('numpy: ', a)
print('tensorflow: ', b)
