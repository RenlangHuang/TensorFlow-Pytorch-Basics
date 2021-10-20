import tensorflow as tf

tensorflow_version = tf.__version__
gpu_available = tf.test.is_gpu_available()

print("tensorflow version:", tensorflow_version, "\tGPU available:", gpu_available)
print('-----------------------------')
tf.config.list_physical_devices('GPU')

a = tf.constant([1.0, 2.0])
b = tf.constant([1.0, 2.0])
result = tf.add(a, b, name="add")
print(result)
