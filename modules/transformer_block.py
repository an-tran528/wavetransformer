#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import Tensor
from torch.nn import Module, GRU, Linear, Dropout, MultiheadAttention, LayerNorm
from .sublayer_connection import SublayerConnection
import numpy as np
import torch
import math, copy
import torch.nn as nn
import torch.nn.functional as F

__author__ = 'An Tran'
__docformat__ = 'reStructuredText'
__all__ = ['TransformerBlock']


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    

class TransformerBlock(Module):

    def __init__(self,
                 n_features: int,
                 n_hidden: int,
                 num_heads: int,
                 nb_classes: int,
                 dropout_p: float) \
                -> None:
        """TransformerBlock decoder.

        """
        super(TransformerBlock, self).__init__()

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.num_heads = num_heads
        
        
        self.mha_1 : Module = MultiheadAttention(
            embed_dim = n_features,
            num_heads = num_heads,
            dropout = dropout_p 
        )
        self.mha_2 : Module = MultiheadAttention(
            embed_dim = n_features,
            num_heads = num_heads,
            dropout = dropout_p 
        )
        
        self.sublayers = clones(SublayerConnection(n_features, dropout_p), 3)

        self.feed_forward = nn.Sequential(
            Linear(n_features, n_hidden),
            nn.ReLU(),
            Dropout(dropout_p),
            Linear(n_hidden, n_features),
        )
    def forward(self,
                word_embs: Tensor,
                encoder_output: Tensor,
                attention_mask: Tensor,
                subsequent_mask: Tensor)\
                -> Tensor:
        x = self.sublayers[0](word_embs, lambda word_embs: self.mha_1(word_embs, word_embs, word_embs, attn_mask=subsequent_mask)[0])
        y = self.sublayers[1](x, lambda x: self.mha_2(x, encoder_output, encoder_output )[0])
        
        out = self.sublayers[2](y, self.feed_forward)        
        return out
            
# EOF