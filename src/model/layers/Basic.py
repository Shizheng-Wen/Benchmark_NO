import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from einops import rearrange, repeat

ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU(0.1),
    'softplus': nn.Softplus,
    'ELU': nn.ELU,
    'silu': nn.SiLU
}

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Fill a tensor with samples from a truncated normal distribution using PyTorch.
    
    Args:
        tensor: an n-dimensional torch.Tensor
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: minimum truncation value (in standard deviations)
        b: maximum truncation value (in standard deviations)
    """
    # Define the bounds for the truncated normal distribution
    init.normal_(tensor, mean=mean, std=std)
    
   
    low = mean + a * std
    high = mean + b * std
    new_tensor = torch.normal(mean=mean, std=std, size=tensor.size(), device=tensor.device)
    new_tensor = torch.clamp(new_tensor, min=low, max=high)
    tensor.data.copy_(new_tensor)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class LinearAttention(nn.Module):
    """
    modified from https://github.com/HaoZhongkai/GNOT/blob/master/models/mmgpt.py
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., attn_type='l1', **kwargs):
        super(LinearAttention, self).__init__()
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        # regularization
        self.attn_drop = nn.Dropout(dropout)
        # output projection
        self.proj = nn.Linear(dim, dim)
        self.n_head = heads
        self.dim_head = dim_head
        self.attn_type = attn_type

    def forward(self, x, y=None):
        y = x if y is None else y
        B, T1, C = x.size()
        _, T2, _ = y.size()
        q = self.query(x).view(B, T1, self.n_head, self.dim_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2)  # (B, nh, T, hs)

        if self.attn_type == 'l1':
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)  # normalized
        elif self.attn_type == "galerkin":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            D_inv = 1. / T2
        elif self.attn_type == "l2":  # still use l1 normalization
            q = q / q.norm(dim=-1, keepdim=True, p=1)
            k = k / k.norm(dim=-1, keepdim=True, p=1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).abs().sum(dim=-1, keepdim=True)  # normalized
        else:
            raise NotImplementedError

        context = k.transpose(-2, -1) @ v
        y = self.attn_drop((q @ context) * D_inv + q)

        # output projection
        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.proj(y)
        return y
