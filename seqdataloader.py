from vocab import Vocab
import random
import re
from d2l import torch as d2l
import torch

def read_time_machine():
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误: 未知词元类型: ' + token)

def load_corpus_time_machine(max_tokens=-1):
    d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7a70c295757f5d63cboa180b6961891a')
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)
    
    def data(pos):
        return corpus[pos: pos + num_steps]
    
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y

class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        if use_random_iter:
            #self.data_iter_fn = seq_data_iter_random(self.corpus, batch_size, num_steps)
            self.data_iter_fn = seq_data_iter_random
        else:
            #self.data_iter_fn = seq_data_iter_sequential(self.corpus, batch_size, num_steps)
            self.data_iter_fn = seq_data_iter_sequential
        self.batch_size = batch_size
        self.num_steps = num_steps
        
    def __iter__(self):
        return self.data_iter_fn(self.corpus,
                                 self.batch_size,
                                 self.num_steps)
            
def load_data_time_machine(batch_size, num_steps, use_random_iter = False, max_tokens = 10000):
    
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    
    return data_iter, data_iter.vocab