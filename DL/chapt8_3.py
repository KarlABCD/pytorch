import torch
from d2l import torch as d2l

X = torch.normal(0, 1, (3, 1))
W_xh = torch.normal(0, 1, (1, 4))
H = torch.normal(0, 1, (3, 4))
W_hh = torch.normal(0, 1, (4, 4))
print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))
print(torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))