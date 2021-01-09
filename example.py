import torch
from torch import nn

from tab_transformer_pytorch import TabTransformer

cont_mean_std = torch.randn(10, 2)

model = TabTransformer(
    categories=(
        10,
        5,
        6,
        5,
        8,
    ),  # tuple containing the number of unique values within each category
    num_continuous=10,  # number of continuous values
    dim=32,  # dimension, paper set at 32
    dim_out=1,  # binary prediction, but could be anything
    depth=6,  # depth, paper recommended 6
    heads=8,  # heads, paper recommends 8
    attn_dropout=0.1,  # post-attention dropout
    ff_dropout=0.1,  # feed forward dropout
    mlp_hidden_mults=(
        4,
        2,
    ),  # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act=nn.ReLU(),  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    continuous_mean_std=cont_mean_std,  # (optional) - normalize the continuous values before layer norm
)

x_categ = torch.randint(
    0, 5, (1, 5)
)  # category values, from 0 - max number of categories, in the order as passed into the constructor above
x_cont = torch.randn(
    1, 10
)  # assume continuous values are already normalized individually

pred = model(x_categ, x_cont)
