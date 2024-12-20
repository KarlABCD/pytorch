import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torchvision
from torch.utils import data
from torchvision import transforms
from accumulator import Accumulator
from animator import Animator
from BN import BatchNorm

class Conv2dSig(nn.Module):
    def __init__(self, inputs, outputs, kernel_size, kernel_size_pool, padding = 0, stride = 1, padding_pool = 0, stride_pool = 1):
        super(Conv2dSig, self).__init__()
        self.conv = nn.Conv2d(in_channels= inputs,
                         out_channels = outputs,
                         kernel_size = kernel_size,
                         padding = padding,
                         stride = stride)
        self.bn = BatchNorm(outputs, num_dims=4)
        self.sigmoid = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size_pool,
                                 padding=padding_pool,
                                 stride=stride_pool)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        x = self.pool(x)
        return x

class LinearSig(nn.Module):
    def __init__(self, inputs, outputs):
        super(LinearSig, self).__init__()
        self.linear = nn.Linear(in_features=inputs, out_features=outputs)
        self.bn = BatchNorm(outputs, num_dims=2)
        self.sigmoid = nn.ReLU()

    def forward(self,x) -> Tensor:
        x = self.linear(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        return x

class LeNet5(nn.Module):
    def __init__(self) -> None:
        super(LeNet5, self).__init__()
        self.Conv2dSig1 = Conv2dSig(1, 6, kernel_size=5, padding=2, kernel_size_pool=2, stride_pool=2)
        self.Conv2dSig2 = Conv2dSig(6, 16, kernel_size=5, kernel_size_pool=2, stride_pool=2)
        self.LinearSig1 = LinearSig(16 * 5 * 5, 120)
        self.LinearSig2 = LinearSig(120, 84)
        self.Flatten = nn.Flatten()
        self.Output = nn.Linear(84, 10)
        
    def forward(self, x) -> Tensor:
        x = self.Conv2dSig1(x)
        x = self.Conv2dSig2(x)
        x = self.Flatten(x)
        x = self.LinearSig1(x)
        x = self.LinearSig2(x)
        x = self.Output(x)
        return x
    
def load_data_fashion_mnist(batch_size, resize=None) -> data:
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root = "../data", train = True, transform = trans, download = False)
    mnist_test = torchvision.datasets.FashionMNIST(root = "../data", train = False, transform = trans, download = False)
    return data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers = 4), data.DataLoader( mnist_test, batch_size, shuffle=True, num_workers = 4)

def accuracy(y_hat, y_target):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y_target.dtype) == y_target
    return float(cmp.type(y_target.dtype).sum())

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

if __name__ == '__main__':
    batch_size = 256
    lr = 0.9
    epochs = 5
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
    model = LeNet5()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr)
    metric = Accumulator(3)
    animator = Animator(xlabel = 'epoch',
                        xlim=[1, epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    num_batchers = len(train_iter)
    for epoch in range(epochs):
        for i, (x, y_target) in enumerate(train_iter):
            optimizer.zero_grad()
            y = model(x)
            yloss = loss(y, y_target)
            yloss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(yloss * x.shape[0], accuracy(y, y_target), x.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2] 
            if( i + 1 ) % (num_batchers // 5) == 0 or i == num_batchers - 1:
                animator.add(epoch + (i + 1) / num_batchers, 
                                (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(model, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, ', f'test acc {test_acc:.3f}')