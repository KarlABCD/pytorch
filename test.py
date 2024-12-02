from timer import Timer
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import numpy as np
#import matplotlib.pyplot as plt
from d2l import torch as d2l
from accumulator import Accumulator
from animator import Animator

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize = figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def get_dataloader_workers():
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root = "../data", train = True, transform = trans, download = False)
    mnist_test = torchvision.datasets.FashionMNIST(root = "../data", train = False, transform = trans, download = False)
    return data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers = get_dataloader_workers()), data.DataLoader( mnist_test, batch_size, shuffle=True, num_workers = get_dataloader_workers())

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim = True)
    return X_exp / partition

def cross_entropy(y_hat, y):
    loss = -torch.log(y_hat[range(len(y_hat)),y])
    return loss

def net(X, W, b):
    y = softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)
    return y

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter, W, b):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in data_iter:
            metric.add(accuracy(net(x, W, b), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater, W, b):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(x, W, b)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(x.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, W, b):
    animator = Animator(xlabel='epoch',
                        xlim=[1,num_epochs],
                        ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater, W, b)
        test_acc = evaluate_accuracy(net, test_iter, W, b)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    #assert train_loss < 0.5, train_loss
    #assert train_acc <= 1 and train_acc > 0.7, train_acc
    #assert test_acc <= 1 and test_acc > 0.7, test_acc
    
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
    
def predict_ch3(net, test_iter, W, b, n=6):
    for x, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(x, W, b).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(x[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    
if __name__ == '__main__':
    d2l.use_svg_display()
    batch_size = 256
    num_inputs = 784
    num_outputs = 10
    lr = 0.1
    W = torch.normal(0, 0.01, size = (num_inputs, num_outputs), requires_grad = True)
    b = torch.zeros(num_outputs, requires_grad = True)
    num_epochs = 5
    train_iter, test_iter = load_data_fashion_mnist(32)
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater, W, b)
    #d2l.plt.show()
    predict_ch3(net, test_iter, W, b)
    d2l.plt.show()
    #x, y = next(iter(data.DataLoader(mnist_train, batch_size = 18)))
    #for x, y in train_iter:
    #    print(x.shape, x.dtype, y.shape, y.dtype)
    #    break
    #show_images(x.reshape(18, 28, 28), 2, 9, titles = get_fashion_mnist_labels(y))