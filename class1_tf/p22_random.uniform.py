import tensorflow as tf

d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print("d:", d)
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print("e:", e) # 2σ

f = tf.random.uniform([2, 2], minval=0, maxval=1)
print("f:", f)