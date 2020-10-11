#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional
import math

import torch
from torch import nn

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['PositionalEncoding']


class PositionalEncoding(nn.Module):

    def __init__(self,
                 d_model: int,
                 dropout: Optional[float] = 0.1,
                 max_len: Optional[int] = 5000) \
            -> None:
        """Positional encoding. Code from:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        :param d_model: Input dimensionality
        :type d_model: int
        :param dropout: Dropout for the encoding, defaults to 0.1.
        :type dropout: float, optional
        :param max_len: Maximum sequence length, defaults to 5000.
        :type max_len: int, optional
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self,
                x: torch.Tensor) \
            -> torch.Tensor:
        """Forward passing.

        :param x: Input.
        :type x: torch.Tensor
        :return: Output.
        :rtype: torch.Tensor
        """
        to_add = self.pe[:, :x.size(1), :].expand(x.size()[0], -1, -1)
        to_add.requires_grad_(False)
        return self.dropout(x + to_add)

# EOF
