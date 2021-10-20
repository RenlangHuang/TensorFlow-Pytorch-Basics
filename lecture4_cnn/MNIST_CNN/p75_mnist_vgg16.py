import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPool2D,
    Dropout,
    Flatten,
    Dense
)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape", x_train.shape)
print("y_train.shape", y_train.shape)
print("x_test.shape", x_test.shape)
print("y_test.shape", y_test.shape)
print("categories:",set(y_train))
np.set_printoptions(precision=2)

# 可视化训练集输入
sample = np.random.randint(0,x_train.shape[0])
plt.imshow(x_train[sample], cmap='gray') # 绘制灰度图
print('the label of the sample:',y_train[sample])
plt.show()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) # 给数据增加一个维度，使数据和网络结构匹配
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print("x_train.shape", x_train.shape)


model = tf.keras.models.Sequential([
    Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    # 2 conv
    Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    Dropout(0.2),
    # 3 conv
    Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    # 4 conv
    Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    Dropout(0.2),
    # 5 conv
    Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    # 6 conv
    Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    # 7 conv
    Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    Dropout(0.2),
    # 8 conv
    Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    # 9 conv
    Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    # 10 conv
    Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    Dropout(0.2),
    # 11 conv
    Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    # 12 conv
    Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    # 13 conv
    Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    Dropout(0.2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.002),#'adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/VGG16_mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

'''cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=4, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

file = open('./weights_mnist_vgg16.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
file.close()

# 显示训练集和验证集的acc和loss曲线
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
plt.show()'''

# prediction
image = tf.reshape(x_train[sample],(1,28,28,1))
image = tf.cast(image,dtype=tf.float32)
print(image.shape)
result = model.predict(image)
print('softmax: ',result)
print('softmax sum =',tf.reduce_sum(result))
print('pred =',tf.argmax(result[0]).numpy(),', real =',y_train[sample])