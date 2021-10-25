import tensorflow as tf

def evaluate(model,inputs,labels,batch_size=32):
    accuracy = 0
    for i in range(0,inputs.shape[0],batch_size):
        if i+batch_size>=inputs.shape[0]:
            batch = inputs[i:]
        else: batch = inputs[i:i+batch_size]
        outputs = model(batch)
        outputs = tf.argmax(outputs,axis=1)
        for k in range(outputs.shape[0]):
            if outputs[k]==labels[k+i]:
                accuracy += 1
    return float(accuracy)/float(inputs.shape[0])*100.0
