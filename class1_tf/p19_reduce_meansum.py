import tensorflow as tf

x = tf.constant([[1, 2, 3], [2, 2, 3]])
print("x:", x)
print("mean of x:", tf.reduce_mean(x))  # 求x中所有数的均值
print("sum of x:", tf.reduce_sum(x))  # 求x中所有数的和
print("sum of x:", tf.reduce_sum(x, axis=0))  # 求每一列的和
print("sum of x:", tf.reduce_sum(x, axis=1))  # 求每一行的和

print(tf.stack([x,x]))
print(tf.stack([x,x],axis=0))
print(tf.stack([x,x],axis=1))
print(tf.stack([x,x],axis=2))

print('concat 1: ',tf.concat([x,x],axis=0))
print('concat 2: ',tf.concat([x,x],axis=1))
