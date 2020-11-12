#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import cat, Tensor
from torch.nn import Module, GRU, Dropout, ModuleList, Conv1d
import torch.nn as nn
import torch
__author__ = 'An Tran'
__docformat__ = 'reStructuredText'
__all__ = ['WaveBlock']


class WaveBlock(Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilation_rates: float,
                 kernel_size: int) \
            -> None:
        """WaveBlock module.
        Adapted from:
        https://www.kaggle.com/c/liverpool-ion-switching/discussion/145256
        """
        super(WaveBlock, self).__init__()

        self.num_rates: float = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(Conv1d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1))

        dilation_rates = [2**i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=int((dilation_rate*(kernel_size-1))/2),
                    dilation=dilation_rate,
                )
            )

            self.gate_convs.append(
                Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=int((dilation_rate*(kernel_size-1))/2),
                    dilation=dilation_rate,
                )
            )

            self.convs.append(
                Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                )
            )
        """ 
        for i in range(len(self.convs)):
            nn.init.xavier_uniform_(self.convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.convs[i].bias)

        for i in range(len(self.filter_convs)):
            nn.init.xavier_uniform_(self.filter_convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.filter_convs[i].bias)

        for i in range(len(self.gate_convs)):
            nn.init.xavier_uniform_(self.gate_convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.gate_convs[i].bias)
        """

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the encoder.
        TODO: update description
        :param x: Input to the encoder.
        :type x: torch.Tensor
        :return: Output of the encoder.
        :rtype: torch.Tensor
        """

        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](
                x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i+1](x)
            res = res + x
        return res

# EOF
