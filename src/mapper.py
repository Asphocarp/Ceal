import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import *


class MappingNetwork(nn.Module):

    def __init__(
            self,
            bit_length: int,  # in
            total_size: int,  # out: maybe make it a list, a output size for each head
            hidden_dims: list,  # like [1024, 1024]
            bias=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.bit_length = bit_length
        self.total_size = total_size

        dd_kwargs = {  # kwargs about device and dtype
            'device': device,
            'dtype': dtype,
        }

        # >> not using EqualLinear
        seq_list = []
        cur_dim = self.bit_length
        for dim in hidden_dims:
            seq_list.append(nn.Linear(cur_dim, dim, bias=bias, **dd_kwargs))
            seq_list.append(nn.ReLU())
            cur_dim = dim
        if len(hidden_dims) == 0:
            cur_dim = self.bit_length
        seq_list.append(nn.Linear(cur_dim, self.total_size, bias=bias, **dd_kwargs))
        self.seq = torch.nn.Sequential(*seq_list)

    def forward(self, msg) -> Tensor:
        # from 0,1 to -1,1 (norm msg at input)
        msg = msg * 2 - 1
        return self.seq(msg)

