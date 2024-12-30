import math
import torch
from torch import nn
from torch.nn import functional as F
import seqdataloader
#from RNN import RNNModelSctratch
from d2l import torch as d2l
import RNN

def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda:torch.tensor([outputs[-1]],device=device).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join(vocab.idx_to_token[i] for i in outputs)

def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [ p for p in net. parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
            
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state = None
    timer = d2l.Timer()
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size = X.shape[0], device = device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X = X.to(device)
        y = y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size = 1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net,
              train_iter,
              vocab,
              lr,
              num_epochs,
              device,
              use_random_iter = False):
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch',
                            ylabel= 'perplexity',
                            legend = ['train'],
                            xlim = [10, num_epochs])
    
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, 
                                     train_iter,
                                     loss,
                                     updater,
                                     device,
                                     use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
        
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('timer traveller'))
    print(predict('traveller'))

batch_size = 32
num_steps = 35
train_iter, vocab = seqdataloader.load_data_time_machine(batch_size, num_steps)
#F.one_hot(torch.tensor([0 , 2]), len(vocab))
#X = torch.arange(10).reshape((2 , 5))
#F.one_hot(X.T, 28).shape

num_hiddens = 512
net = RNN.RNNModelSctratch(len(vocab),
                            num_hiddens,
                            d2l.try_gpu(),
                            RNN.get_params,
                            RNN.init_rnn_state,
                            RNN.rnn)
#state = net.begin_state(X.shape[0], d2l.try_gpu())
#Y, new_state = net(X.to(d2l.try_gpu), state)
#print(Y.shape, len(new_state), new_state[0].shape)

num_epochs = 500
lr = 1

train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())