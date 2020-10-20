#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import cat, Tensor
from torch.nn import Module, Linear, GRU, ModuleList, ReLU, BatchNorm1d, Conv1d, Sequential, BatchNorm2d, Conv2d
from .wave_block import WaveBlock
from typing import List
from .depthwise_separable_conv_block import DepthWiseSeparableConvBlock
from .dessed_dnn import DepthWiseSeparableDNN
import torch
import torch.nn.functional as F
__author__ = 'An Tran'
__docformat__ = 'reStructuredText'
__all__ = ['DessedDNNEncoder']


class DessedDNNEncoder(Module):

    def __init__(self,
                 in_channels: int,
                 cnn_channels: int,
                 inner_kernel_size: int,
                 inner_padding: int,
                 last_dim: int,
                 ) \
            -> None:
        """DessedDNNEncoder module.
        :param in_channels: Input channels.
        :type in_channels: int
        :param cnn_channels: Amount of output CNN channels.
        :type cnn_channels: int
        :param inner_kernel_size: Kernel shape/size of the second convolution\
                                  for DWS-DNN.
        :type inner_kernel_size: int
        :param inner_padding: Inner padding.
        :type inner_padding: int
        """
        super(DessedDNNEncoder, self).__init__()

        self.in_channels: int = in_channels
        self.cnn_channels: int = cnn_channels
        self.dnn = DepthWiseSeparableDNN(
            cnn_channels=cnn_channels,
            cnn_dropout=0.2,
            inner_kernel_size=inner_kernel_size,
            inner_padding=inner_padding
        )
        
        self.fc_audioset = Linear(last_dim, last_dim, bias=True)
        
    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the encoder.

        :param x: Input to the encoder.
        :type x: torch.Tensor
        :return: Output of the encoder.
        :rtype: torch.Tensor
        """
        bs, time_step, mel_bins = x.size()
        # CNN
        x = self.dnn(x)
        x = x.transpose(1,3) # (bs, 1, time_steps, 128)

        x = x.squeeze(1)
        x = F.relu_(self.fc_audioset(x))
        return x

# EOF
