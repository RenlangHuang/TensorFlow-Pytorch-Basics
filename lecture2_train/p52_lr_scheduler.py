import numpy as np
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
import tensorflow.keras.optimizers.schedules as schedulers

def learning_rate(scheduler,steps):
    learning_rates = []
    for i in range(steps):
        learning_rates.append(scheduler(i))
    return learning_rates

lr_scheduler = schedulers.PiecewiseConstantDecay(boundaries=[30,50,70,90],values=[1.0,0.5,0.2,0.1,0.05])
lr = learning_rate(lr_scheduler,100)
plt.plot(lr,label='PiecewiseConstantDecay')

lr_scheduler = schedulers.InverseTimeDecay(1.0,1,0.01) #lr=lr0/(1+decay_rate*t/T)
lr = learning_rate(lr_scheduler,100)
plt.plot(lr,label='InverseTimeDecay')

lr_scheduler = schedulers.ExponentialDecay(1.0,100,0.1) #lr=lr0*rate**(t/T)
lr = learning_rate(lr_scheduler,100)
plt.plot(lr,label='ExponentialDecay')

lr_scheduler = tf.keras.experimental.CosineDecay(1.0,100) #
lr = learning_rate(lr_scheduler,100)
plt.plot(lr,label='CosineDecay')

# learning rate scheduler can be defined by ourselves
class NaturalExponentialDecay(schedulers.LearningRateSchedule):
    def __init__(self,initial_learning_rate,decay_steps,decay_rate) -> None:
        super().__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate,dtype=tf.float32)
        self.decay_steps = tf.cast(decay_steps, dtype=tf.float32)
        self.decay_rate = tf.cast(decay_rate, dtype=tf.float32)
    def __call__(self, step):
        #return super().__call__(step)
        return self.initial_learning_rate*tf.math.exp(-self.decay_rate * step / self.decay_steps)

lr_scheduler = NaturalExponentialDecay(1.0,100,1.0)
lr = learning_rate(lr_scheduler,100)
plt.plot(lr,label='NaturalExponentialDecay')

plt.legend()
plt.show()


# using learning rate scheduler in training:
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

model = tf.keras.models.Sequential([
    Dense(4,activation='relu',name='hidden'),
    Dense(3,activation='softmax',name='classification')
])

initial_learning_rate = 0.005
lr_schedule = schedulers.ExponentialDecay(initial_learning_rate, 500, 0.8)

model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy'])

history = model.fit(x_train, y_train, batch_size=32, epochs=500, validation_data=(x_test, y_test), validation_freq=1)
model.summary()


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