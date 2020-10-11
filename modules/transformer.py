import torch
import torch.nn as nn
from .transformer_block import TransformerBlock,clones
from torch.nn import Module, LayerNorm
from torch import Tensor
from .positional_encoding import PositionalEncoding
import numpy as np

def subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Transformer(Module):
    def __init__(self, layer, num_layers, 
                 nb_classes, n_features,dropout_p):
        super(Transformer, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.n_features)
        self.positional_encoding = PositionalEncoding(n_features,dropout_p)
        self.num_layers = num_layers

    def forward(self,
                word_embed: Tensor,
                encoder_output: Tensor,
                attention_mask: Tensor,
                ) \
                -> Tensor:
        device = word_embed.device
        mask_size = word_embed.size()[0]
        s_mask = subsequent_mask(mask_size).to(device)
        y = self.positional_encoding(word_embed)

        for i in range(self.num_layers):
            y = self.layers[i](y, encoder_output, attention_mask, s_mask)
        return self.norm(y)