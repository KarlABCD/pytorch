import torch
import torch.nn as nn
from d2l import torch as d2l

def init_params():
    w = torch.normal(0, 1, size = (num_inputs, 1), requires_grad = True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2) / 2)

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs = 100
    lr = 0.01
    animator = d2l.Animator(xlabel='epochs',
                            ylabel='loss',
                            yscale= 'log',
                            xlim = [5, num_epochs],
                            legend = ['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch % 5) == 0:
            animator.add(epoch + 1,(d2l.evaluate_loss(net, train_iter, loss),
                                    d2l.evaluate_loss(net, test_iter, loss)))
        print('W的L2范数是: ', torch.norm(w).item())
            

n_train = 20
n_test = 100
num_inputs = 200
batch_size = 5

true_w = torch.ones((num_inputs, 1)) * 0.01 
true_b = 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train = False)
train(lambd=1)
d2l.plt.show()