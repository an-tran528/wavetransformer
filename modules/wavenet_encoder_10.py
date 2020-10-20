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
__all__ = ['WaveNetEncoder10']


class WaveNetEncoder10(Module):

    def __init__(self,
                 in_channels: int,
                 out_waveblock: List,
                 kernel_size: int,
                 dilation_rates: List,
                 inner_kernel_size: int,
                 inner_padding: int,
                 last_dim: int,
                 pw_kernel: int,
                 pw_padding:int,
                 merge_mode: str,
                 ) \
            -> None:
        """WaveNetEncoder10 module (1D+2D CNN).
        :param in_channels: Input channels.
        :type in_channels: int
        :param out_waveblock: Output channels for the wave blocks
        :type out_waveblock: List
        :param kernel_size: Kernel shape/size for the wave blocks
        :type kernel_size: List
        :param dilation_rates: Dilation factors for the wave blocks
        :type dilation_rates: List
        :param inner_kernel_size: Kernel size for DWS-DNN
        :type inner_kernel_size: int
        :param inner_padding: Inner padding for DWS-DNN
        :type inner_padding: int
        :param pw_kernel: Kernel size for the merging Conv2d
        :type pw_kernel: int
        :param pw_padding: padding for the merging Conv2d
        :type pw_padding: int
        :param merge_mode: merging mode (conv/mean)
        :type merge_mode: str
        :param last_dim: Output channels for Linear layer.
        :type last_dim: int
        """
        super(WaveNetEncoder10, self).__init__()

        assert len(dilation_rates) == len(out_waveblock), "length of dilation rates must match out channels"
        self.in_channels: int = in_channels
        self.out_waveblock = out_waveblock
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.pw_kernel =pw_kernel
        self.pw_padding =pw_padding

        self.wave_blocks = ModuleList() 
        self.bn_relu = ModuleList()
        self.wave_blocks.append(WaveBlock(
            in_channels=self.in_channels,
            out_channels=self.out_waveblock[0],
            dilation_rates=self.dilation_rates[0],
            kernel_size=self.kernel_size
        ))
        
        for i in range(1,len(self.dilation_rates)):
            self.wave_blocks.append(
                WaveBlock(
                    in_channels=self.out_waveblock[i-1],
                    out_channels=self.out_waveblock[i],
                    dilation_rates=self.dilation_rates[i],
                    kernel_size=self.kernel_size
                )
            )
        
        for i in range(len(self.dilation_rates)):
            self.bn_relu.append(
                Sequential(
                    BatchNorm1d(self.out_waveblock[i]),
                    ReLU(),
                )
            )

        self.dnn = DepthWiseSeparableDNN(
            cnn_channels=out_waveblock[-1],
            cnn_dropout=0.2,
            inner_kernel_size=inner_kernel_size,
            inner_padding=inner_padding
        )
        if merge_mode == 'conv':
            self.point_wise: Module = Conv2d(
                in_channels=2, out_channels=1,
                kernel_size=self.pw_kernel, stride=1,
                padding=self.pw_padding, dilation=1,
                groups=1, bias=True, padding_mode='zeros')
        elif merge_mode == 'mean':
            self.point_wise = None
        else:
            raise Exception('Incorrect argument')
        self.fc_audioset = Linear(last_dim,last_dim, bias=True)

        
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
        x1 = x.permute(0,2,1)
        x2 = x

        # WaveNet
        for i in range(len(self.wave_blocks)):
            x1 = self.wave_blocks[i](x1)
            x1 = self.bn_relu[i](x1) 
        
        x1 = x1.unsqueeze(1).transpose(2,3)
        # CNN
        x2 = self.dnn(x2)
        x2 = x2.transpose(1,3) # (bs, 1, time_steps, 128)
        # x2 = x2.unsqueeze(1)
        x = torch.cat((x2, x1), dim=1)
        if self.point_wise != None:
            x = self.point_wise(x) 
            x = x.squeeze(1)
        else: 
            x = x.mean(dim=1)
        x = F.relu_(self.fc_audioset(x))
        return x

# EOF
