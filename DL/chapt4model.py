import torch
import torch.nn as nn
from d2l import torch as d2l
from accumulator import Accumulator
from animator import Animator

num_inputs = 784
num_outputs = 10
num_hidden1 = 256
num_hidden2 = 256
dropout1 = 0.5
dropout2 = 0.2
class Net(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 num_hidden1,
                 num_hidden2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hidden1)
        self.lin2 = nn.Linear(num_hidden1, num_hidden2)
        self.lin3 = nn.Linear(num_hidden2, num_outputs)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        H1 = self.relu(self.lin1(x.reshape(-1, num_inputs)))
        if self.training == True:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out
    
def dropout_layer(x, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(x)
    if dropout == 0:
        return x
    mask = (torch.rand(x.shape) > dropout).float()
    return mask * x /( 1.0 - dropout )
    
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in data_iter:
            metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(x)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(x.shape[0])
        #print("%.7f" %W[1][1])
        #print("%.3f" %i)
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1], metric[1] / metric[2]    
    
    
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch',
                        xlim=[1,num_epochs],
                        ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    #assert train_loss < 0.5, train_loss
    #assert train_acc <= 1 and train_acc > 0.7, train_acc
    #assert test_acc <= 1 and test_acc > 0.7, test_acc

def predict_ch3(net, test_iter, n=6):
    for x, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(x).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(x[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

if __name__ == '__main__':

    net = Net(num_inputs, num_outputs, num_hidden1, num_hidden2)
    num_epochs = 10
    lr = 0.5
    batch_size = 256
    loss = nn.CrossEntropyLoss(reduction = 'none')
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    predict_ch3(net, test_iter)
    d2l.plt.show()