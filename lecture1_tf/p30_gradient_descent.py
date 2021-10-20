import tensorflow as tf
import matplotlib.pyplot as plt

def train(w,Loss,lr=0.01,epochs=50,optimizer='SGD',display=5,beta=None):
    mt,Vt = 0,0
    losses = []
    for epoch in range(epochs):
        with tf.GradientTape() as tape: # with结构到grad框起了梯度的计算过程
            loss = Loss(w)
        losses.append(loss)
        grad = tape.gradient(loss,w) # .gradient函数告知谁对谁求导
        if optimizer == 'SGD':
            w.assign_sub(lr * grad) # .assign_sub 对变量做自减
        elif optimizer == 'SGDM':
            if beta is None: beta = 0.9
            mt = beta*mt + (1.0-beta)*grad
            w.assign_sub(lr * mt)
        elif optimizer == 'Adagrad':
            Vt = Vt+tf.square(grad)
            w.assign_sub(lr * grad / tf.sqrt(Vt))
        elif optimizer == 'RMSProp':
            if beta is None: beta = 0.2
            Vt = beta*Vt + (1-beta)*tf.square(grad)
            w.assign_sub(lr * grad / tf.sqrt(Vt))
        elif optimizer == 'Adam':
            if beta is None: beta = (0.2,0.2)
            mt = beta[0]*mt+(1-beta[0])*grad
            Vt = beta[1]*Vt+(1-beta[1])*tf.square(grad)
            mt_ = mt/(1.0-tf.pow(beta[0],epoch+1))
            Vt_ = Vt/(1.0-tf.pow(beta[1],epoch+1))
            w.assign_sub(lr * mt_ / tf.sqrt(Vt_))
        if display is None: continue
        if (epoch+1) % display == 0:
            print("After %s epochs,w is %f,loss is %f" % (epoch+1, w.numpy(), loss))
    print("After %s epochs,w is %f,loss is %f" % (epochs, w.numpy(), loss))
    return w, losses


Loss = lambda w: tf.square(w + 1)

w = tf.Variable(tf.constant(5, dtype=tf.float32))
_,losses = train(w,Loss,lr=0.05,epochs=40)
plt.plot(losses,label='SGD, lr=0.05')

w = tf.Variable(tf.constant(5, dtype=tf.float32))
_,losses = train(w,Loss,lr=0.2,epochs=40,optimizer='SGDM')
plt.plot(losses,label='SGDM, lr=0.2, beta=0.9')

w = tf.Variable(tf.constant(5, dtype=tf.float32))
_,losses = train(w,Loss,lr=0.6,epochs=40,optimizer='Adagrad')
plt.plot(losses,label='Adagrad, lr=0.6')

w = tf.Variable(tf.constant(5, dtype=tf.float32))
_,losses = train(w,Loss,lr=0.2,epochs=40,optimizer='RMSProp')
plt.plot(losses,label='RMSProp, lr=0.2')

w = tf.Variable(tf.constant(5, dtype=tf.float32))
_,losses = train(w,Loss,lr=0.2,epochs=40,optimizer='Adam')
plt.plot(losses,label='Adam, lr=0.2')

plt.legend()
plt.grid()
plt.show()

# 最终目的：找到 loss 最小 即 w = -1 的最优参数w
