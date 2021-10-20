import numpy as np
import tensorflow as tf

classes = 3
labels = tf.constant([1, 0, 2])  # 输入的元素值最小为0，最大为2
output = tf.one_hot(labels, depth=classes)
print("result of labels:", output)

labels = np.arange(6)
np.random.shuffle(labels)
print('labels: ',labels)
labels = tf.constant(labels)
output = tf.one_hot(labels, depth=6)
print("result of labels:", output)

labels = tf.constant(np.arange(6)+1)
output = tf.one_hot(labels, depth=6)
print("result of labels:", output)

labels = tf.constant(np.arange(6)+1)
output = tf.one_hot(labels, depth=3)
print("result of labels:", output)