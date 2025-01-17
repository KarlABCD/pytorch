import torch 
import torch.nn as nn
from d2l import torch as d2l

def dropout_layer(x, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(x)
    if dropout == 0:
        return x
    mask = (torch.rand(x.shape) > dropout).float()
    return mask * x /( 1.0 - dropout )
x = torch.arange(16, dtype=torch.float32).reshape(2,8)
print(x)
print(dropout_layer(x, 0.))
print(dropout_layer(x, 0.5))
print(dropout_layer(x, 1.))