import tensorflow as tf

a = tf.constant([1, 5], dtype=tf.int64)
print("a:", a)
print("a.dtype:", a.dtype)
print("a.shape:", a.shape)

# 本机默认 tf.int32  可去掉dtype试一下 查看默认值
print(tf.constant([1,-1]).dtype)
print(tf.constant([1.0,-1.0]).dtype)
print(tf.constant([1.0,-0.2]).dtype)
print(tf.constant([1.0,-0.2],dtype=tf.float64))
print(tf.constant("string"), tf.constant("string").dtype)