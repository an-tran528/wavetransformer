#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Module
from torch import Tensor
from typing import Optional, Union, Tuple, List

from torch import Tensor, zeros, cat as torch_cat
from torch.nn import Module, Linear, Softmax, Embedding
from torch.nn.functional import softmax

from modules import WaveBlock, WaveNetEncoder3
from modules.transformer import Transformer
from modules.transformer_block import TransformerBlock
from modules.positional_encoding import PositionalEncoding
import torch
import torch.nn.functional as F
from modules.decode_utils import greedy_decode, topk_sampling
from modules.beam import beam_decode
import gc

__author__ = 'An Tran'
__docformat__ = 'reStructuredText'
__all__ = ['WaveTransformer3']


class WaveTransformer3(Module):
    """
    WaveTransformer full model with only E_temp branch
    (denoted as WT_temp in paper)
    """
    def __init__(self,
                 in_channels_encoder: int,
                 out_channels_encoder: List,
                 kernel_size_encoder: int,
                 dilation_rates_encoder: List,
                 last_dim_encoder: int,
                 num_layers_decoder: int,
                 num_heads_decoder: int,
                 n_features_decoder: int,
                 n_hidden_decoder: int,
                 nb_classes: int,
                 dropout_decoder: float,
                 beam_size: int,
                 ) \
            -> None:
        """WaveTransformer3 model.
        :param in_channels_encoder: Input channels.
        :type in_channels_encoder: int
        :param out_channels_encoder: Output channels for the wave blocks
        :type out_channels_encoder: List
        :param kernel_size_encoder: Kernel shape/size for the wave blocks
        :type kernel_size_encoder: List
        :param dilation_rates_encoder: Dilation factors for the wave blocks
        :type dilation_rates_encoder: List
        :param last_dim_encoder: Output channels for Linear layer
        :type last_dim_encoder: int
        :param num_layers_decoder: Number of transformer blocks
        :type num_layers_decoder: int
        :param num_heads_decoder: Number of attention heads in each MHA
        :type num_heads_decoder: int
        :param n_features_decoder: number of features for transformer
        :type n_features_decoder: int
        :param n_hidden_decoder: hidden dimension of transformer 
        :type n_hidden_decoder: int
        :param nb_classes: vocabulary size 
        :type nb_classes: int
        :param dropout_decoder: dropout rate in decoder
        :type dropout_decoder: float
        :param beam_size: beam size (<1: greedy, >1: beam search) 
        :type beam_size: int
        """
        super(WaveTransformer3, self).__init__()
        self.max_length: int = 22
        self.nb_classes: int = nb_classes
        self.beam_size = beam_size

        self.encoder: Module = WaveNetEncoder3(
            in_channels=in_channels_encoder,
            out_channels=out_channels_encoder,
            kernel_size=kernel_size_encoder,
            dilation_rates=dilation_rates_encoder,
            last_dim=last_dim_encoder)

        self.sublayer_decoder: Module = TransformerBlock(
            n_features=n_features_decoder,
            n_hidden=n_hidden_decoder,
            num_heads=num_heads_decoder,
            nb_classes=nb_classes,
            dropout_p=dropout_decoder
        )

        self.decoder: Module = Transformer(
            layer=self.sublayer_decoder,
            num_layers=num_layers_decoder,
            nb_classes=nb_classes,
            n_features=n_features_decoder,
            dropout_p=dropout_decoder)

        self.embeddings: Embedding = Embedding(
            num_embeddings=nb_classes,
            embedding_dim=n_features_decoder)

        self.classifier: Linear = Linear(
            in_features=n_features_decoder,
            out_features=nb_classes)

    def forward(self, x, y):
        if y is None:
            return self._inference(x)
        else:
            return self._training_pass(x, y)

    def _training_pass(self,
                       x: Tensor,
                       y: Tensor,
                       ) \
            -> Tensor:
        """Forward pass of the baseline method.

        :param x: Input features.
        :type x: torch.Tensor
        :return: Predicted values.
        :rtype: torch.Tensor
        """

        torch.cuda.empty_cache()
        gc.collect()

        b_size, max_len = y.size()
        device = y.device
        y = y.permute(1, 0)[:-1]

        encoder_output: Tensor = self.encoder(x)
        encoder_output = encoder_output.permute(1, 0, 2)
        word_embeddings: Tensor = self.embeddings(y)
        decoder_output: Tensor = self.decoder(
            word_embeddings,
            encoder_output,
            attention_mask=None
        )

        out: Tensor = self.classifier(decoder_output)
        return out

    def _inference(self, x):
        torch.cuda.empty_cache()
        gc.collect()
        eos_token = 9
        if self.beam_size > 1:
            return beam_decode(x,
                               self.encoder,
                               self.decoder,
                               self.embeddings,
                               self.classifier,
                               self.beam_size,
                               1)
        else:
            return greedy_decode(x,
                                 self.encoder,
                                 self.decoder,
                                 self.embeddings,
                                 self.classifier,
                                 self.max_length,
                                 )
# EOF
