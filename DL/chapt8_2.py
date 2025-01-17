from vocab import Vocab
import random
import re
from d2l import torch as d2l
import torch




if __name__ == '__main__':

    #lines = read_time_machine()
    #print(f'# 文本总行数:{len(lines)}')
    #print(lines[0])
    #print(lines[10])
    d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7a70c295757f5d63cboa180b6961891a')
    
    #for i in range(11):
    #    print(tokens[i])
        
    #lines = read_time_machine()
    #tokens = tokenize(lines)
    #vocab = Vocab(tokens)
    #print(vocab.token_freqs[:10])
    #print(list(vocab.token_to_idx.items())[:10])
    #freqs = [freq for token, freq in vocab.token_freqs]
    #d2l.plot(freqs, xlabel = 'token: x', ylabel = 'frequency: n(x)', 
    #         xscale='log', yscale='log')
    #d2l.plt.show()
    
    #corpus = [vocab[token] for line in tokens for token in line]
    #bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    #biagram_vocab = Vocab(bigram_tokens)
    #print(biagram_vocab.token_freqs[:10])
    
    #trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    #trigram_vocab = Vocab(trigram_tokens)
    #print(f'{trigram_vocab.token_freqs[:10]}')
    
    #for i in [0, 10]:
    #    print('文本', tokens[i])
    #    print('索引', vocab[tokens[i]])
    '''my_seq = list(range(35))
    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)'''