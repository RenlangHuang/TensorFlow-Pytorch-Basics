import os
import torch
import numpy as np
from torch.autograd import Variable
from matplotlib import pyplot as plt


def evaluate(model,inputs,labels,batch_size=32):
    accuracy = 0
    model.eval()
    gpu_available = torch.cuda.is_available()
    for i in range(0,inputs.shape[0],batch_size):
        if i+batch_size>=inputs.shape[0]:
            batch = inputs[i:]
        else: batch = inputs[i:i+batch_size]
        if gpu_available:
            batch = batch.cuda()
        with torch.no_grad():
            outputs = model(batch)
        outputs = torch.argmax(outputs,dim=1)
        for k in range(outputs.shape[0]):
            if outputs[k]==labels[k+i]:
                accuracy += 1
    return float(accuracy)/float(inputs.shape[0])*100.0


def training(model, Loss, optimizer, loader, epochs, train, validation,
    model_path, load_pretrained=True, save_model=True, visualizing=True):

    if load_pretrained and os.path.exists(model_path):
        print('------------load the model----------------')
        model.load_state_dict(torch.load(model_path))
    x_train, y_train = train
    x_test, y_test = validation
    losses = []
    accuracy_train = []
    accuracy_test = []
    best_acc = -100.0
    for epoch in range(epochs):
        model.train()
        batch_loss = []
        for step,data in enumerate(loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = Variable(inputs.cuda()),Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs),Variable(labels)
            out = model(inputs)
            loss = Loss(out,labels)
            batch_loss.append(loss.data.item())
            if step % 5 == 4:
                print('epoch %d | batch %d: loss = %.4f'%(epoch+1,step+1,batch_loss[-1]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(np.mean(np.array(batch_loss)))
        if torch.cuda.is_available():
            accuracy_train.append(evaluate(model,x_train.cuda(),y_train.cuda()))
            accuracy_test.append(evaluate(model,x_test.cuda(),y_test.cuda()))
        else:
            accuracy_train.append(evaluate(model,x_train,y_train))
            accuracy_test.append(evaluate(model,x_test,y_test))
        print('epoch: %d: loss = %.4f, mean_loss = %.4f, Training_accuracy = %.4f, Validation_accuracy = %.4f'%\
            (epoch+1,batch_loss[-1],losses[-1],accuracy_train[-1],accuracy_test[-1]))
        print('----------------------------------------------------------------------------')
        current_acc = accuracy_train[-1]+1.5*accuracy_test[-1]-losses[-1]*80
        if current_acc>best_acc: # only save the best model
            best_epoch = epoch
            best_acc = current_acc
            best_model = model.state_dict().copy()

    model.load_state_dict(best_model)
    print('best epoch: %d: Training_accuracy = %.4f, Validation_accuracy = %.4f'%\
        (best_epoch+1,accuracy_train[best_epoch],accuracy_test[best_epoch]))

    # save the model parameters
    if save_model:
        torch.save(model.state_dict(),model_path)

    # visualization
    if visualizing:
        plt.subplot(1,2,1)
        plt.title('training loss')
        plt.plot(losses)
        plt.grid()
        plt.subplot(1,2,2)
        plt.title('model evaluation')
        plt.plot(accuracy_train,label='Training Accuracy')
        plt.plot(accuracy_test,label='Testing Accuracy')
        plt.legend()
        plt.grid()
        plt.show()
