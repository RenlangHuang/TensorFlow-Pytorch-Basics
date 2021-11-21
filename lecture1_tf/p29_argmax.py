import numpy as np
import tensorflow as tf

test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
print("test:\n", test)
print("每一列的最大值的索引：", tf.argmax(test, axis=0))  # 返回每一列最大值的索引
print("每一行的最大值的索引", tf.argmax(test, axis=1))  # 返回每一行最大值的索引

print(test.shape,test.dtype)
print(tf.reshape(test,(2,-1)))
print("每一行的最大值的索引", tf.argmax(test, axis=1))
test = tf.reshape(test,(-1,2))
print("每一列的最大值的索引", tf.argmax(test, axis=0))

# similarly, we can use tf.argmin


# extract data from tensors with indices
data = tf.constant([1, 2, 3, 4, 5, 6])
indices = tf.constant([0, 1, 2, 1])
print(tf.gather(data, indices))

data = tf.constant([[1, 2], [3, 4], [5, 6]])
indices = tf.constant([2, 0, 1])
print(tf.gather(data, indices))

indices = tf.constant([1, 0, 1])
indices = list(enumerate(indices))
print(tf.gather_nd(data, indices))
