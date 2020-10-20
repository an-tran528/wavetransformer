#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import cat, Tensor
from torch.nn import Module, Linear, GRU, ModuleList, ReLU, BatchNorm1d, BatchNorm2d, Sequential
from .wave_block import WaveBlock
from typing import List

__author__ = 'An Tran'
__docformat__ = 'reStructuredText'
__all__ = ['WaveNetEncoder3']


class WaveNetEncoder3(Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: List,
                 kernel_size: int,
                 dilation_rates: List,
                 last_dim: int
                 ) \
            -> None:
        """WaveNetEncoder3 module.
        :param in_channels: Input channels.
        :type in_channels: int
        :param out_channels: Output channels for the wave blocks.
        :type out_channels: List
        :param kernel_size: Kernel shape/size for the wave blocks.
        :type kernel_size: int
        :param dilation_rates: Dilation factors for the wave blocks.
        :type dilation_rates: List
        :param last_dim: Output channels for Linear layer.
        :type last_dim: int
        """
        super(WaveNetEncoder3, self).__init__()

        assert len(dilation_rates) == len(out_channels), "length of dilation rates must match out channels"
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates


        self.wave_blocks = ModuleList() 
        self.bn_relu = ModuleList()
        self.wave_blocks.append(WaveBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels[0],
            dilation_rates=self.dilation_rates[0],
            kernel_size=self.kernel_size
        ))
        
        for i in range(1,len(self.dilation_rates)):
            self.wave_blocks.append(
                WaveBlock(
                    in_channels=self.out_channels[i-1],
                    out_channels=self.out_channels[i],
                    dilation_rates=self.dilation_rates[i],
                    kernel_size=self.kernel_size
                )
            )
        
        for i in range(len(self.dilation_rates)):
            self.bn_relu.append(
                Sequential(
                    BatchNorm1d(self.out_channels[i]),
                    ReLU(),
                )
            )

        self.fc : Module = Linear(self.out_channels[-1],last_dim)

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the encoder.

        :param x: Input to the encoder.
        :type x: torch.Tensor
        :return: Output of the encoder.
        :rtype: torch.Tensor
        """
        x = x.permute(0,2,1)

        for i in range(len(self.wave_blocks)):
            x = self.wave_blocks[i](x)
            x = self.bn_relu[i](x) 

        x = x.permute(0,2,1)
        x = self.fc(x)
        return x

# EOF
