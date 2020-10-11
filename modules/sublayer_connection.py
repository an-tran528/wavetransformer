#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import cat, Tensor
from torch.nn import Module, GRU, Dropout, LayerNorm

__author__ = 'An Tran'
__docformat__ = 'reStructuredText'
__all__ = ['SublayerConnection']


class SublayerConnection(Module):

    def __init__(self,
                 size: int,
                 dropout: float):
        """SublayerConnection module.
        A residule connection followed by a layer norm
        """
        super(SublayerConnection, self).__init__()
        self.norm : Module = LayerNorm(size) 
        self.dropout : Module = Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size """
        return x + self.dropout(sublayer(self.norm(x)))
# EOF
