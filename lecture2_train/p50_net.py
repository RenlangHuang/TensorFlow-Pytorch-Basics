import os, time
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt

# load iris dataset
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target
print('iris dataset:')
print('features:',type(x_data),x_data.shape)
print('labels:',type(y_data),y_data.shape)

# shuffle the dataset randomly
np.random.seed(681)
np.random.shuffle(x_data)
np.random.seed(681)
np.random.shuffle(y_data)
tf.random.set_seed(681)

y_train = y_data[:-30]
y_test = y_data[-30:]
x_train = tf.cast(x_data[:-30], dtype=tf.float32)
x_test = tf.cast(x_data[-30:], dtype=tf.float32)


class SLP(Model):
    def __init__(self):
        super(SLP, self).__init__()
        self.layer = Dense(3,activation='softmax')
    def call(self,x):
        return self.layer(x)


model = SLP()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005),#'adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy'])
checkpoint_pth = './checkpoint/slp.ckpt'
if os.path.exists(checkpoint_pth + '.index'):
    print('-------------load the pre-trained model-------------')
    model.load_weights(checkpoint_pth)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_pth,
                                                 save_weights_only=True,
                                                 save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=500, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()


file = open('./weights_slp.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()