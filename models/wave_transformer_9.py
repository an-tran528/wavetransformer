from torch.nn import Module
from torch import Tensor
from typing import Optional, Union, Tuple, List

from torch import Tensor, zeros, cat as torch_cat
from torch.nn import Module, Linear, Softmax, Embedding, DataParallel
from torch.nn.functional import softmax

from modules import WaveBlock, WaveNetEncoder9
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
__all__ = ['WaveTransformer9']


class WaveTransformer9(Module):

    def __init__(self,
                 in_channels_encoder: int,
                 out_waveblock_encoder: List,
                 kernel_size_encoder: int,
                 dilation_rates_encoder: List,
                 inner_kernel_size_encoder: int,
                 inner_padding_encoder: int,
                 last_dim_encoder: int,
                 num_layers_decoder: int,
                 num_heads_decoder: int,
                 n_features_decoder: int,
                 n_hidden_decoder: int,
                 nb_classes: int,
                 dropout_decoder: float,
                 beam_size: int
                 ) \
            -> None:
        """Baseline method for audio captioning with Clotho dataset.
        TODO: update description
        """
        super(WaveTransformer9, self).__init__()
        self.max_length: int = 22
        self.nb_classes: int = nb_classes
        self.beam_size = beam_size
        self.encoder: Module = WaveNetEncoder9(
            in_channels=in_channels_encoder,
            out_waveblock=out_waveblock_encoder,
            kernel_size=kernel_size_encoder,
            dilation_rates=dilation_rates_encoder,
            inner_kernel_size=inner_kernel_size_encoder,
            inner_padding=inner_padding_encoder,
            last_dim=last_dim_encoder,
            dnn_mode=dnn_mode_encoder)

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
        # torch.cuda.empty_cache()
        # gc.collect()
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
        # torch.cuda.empty_cache()
        # gc.collect()
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
