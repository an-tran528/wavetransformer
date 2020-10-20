#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, MutableSequence, \
    Callable, Optional, List, Union, MutableMapping
from platform import processor
from pathlib import Path
from collections import OrderedDict

from torch import cuda, zeros, Tensor, load as pt_load
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR 
from models import WaveTransformer3, WaveTransformer8, WaveTransformer10
__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_device', 'get_model',
           'module_epoch_passing',
           'module_forward_passing']


def get_device(force_cpu: bool) \
        -> Tuple[str, str]:
    """Gets the available device.

    :param force_cpu: Force CPU usage?
    :type force_cpu: bool
    :return: Device and device name.
    :rtype: str, str
    """
    return ('cuda', cuda.get_device_name(cuda.current_device())) \
        if cuda.is_available() and not force_cpu else \
        ('cpu', processor())


def get_model(settings_model: MutableMapping[str, Union[str, MutableMapping]],
              settings_io: MutableMapping[str, Union[str, MutableMapping]],
              output_classes: int,
              device: str) \
        -> Module:
    """Creates and returns the model for the process.

    :param settings_model: Model specific settings to be used.
    :type settings_model: dict
    :param settings_io: File I/O settings to be used.
    :type settings_io: dict
    :param output_classes: Amount of output classes.
    :type output_classes: int
    :param device: Device to put the loaded model.
    :type device: str
    :return: Model.
    :rtype: torch.nn.Module
    """
    encoder_settings = settings_model['encoder']
    decoder_settings = settings_model['decoder']
    decoder_settings.update({'nb_classes': output_classes})

    kwargs = {**encoder_settings, **decoder_settings}

    model_name = settings_model['model_name']

    if model_name == 'wave_transformer_3':
        model = WaveTransformer3
    elif model_name == 'wave_transformer_8':
        model = WaveTransformer8
    elif model_name == 'wave_transformer_10':
        model = WaveTransformer10
    else:
        raise AttributeError(f'Unknown model type '
                             f'{settings_model["model_name"]}.')

    model = model(**kwargs)

    if settings_model['use_pre_trained_model']:
        state_dict = pt_load(Path(
            settings_io['root_dirs']['outputs'],
            settings_io['model']['model_dir'],
            settings_io['model']['pre_trained_model_name']
        ), map_location=device)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

    return model


def module_epoch_passing(data: DataLoader,
                         module: Module,
                         objective: Union[Callable[[Tensor, Tensor], Tensor], None],
                         optimizer: Union[Optimizer, None],
                         use_y: Optional[bool] = False,
                         grad_norm: Optional[int] = 1,
                         grad_norm_val: Optional[float] = -1.) \
        -> Tuple[Tensor, List[Tensor], List[Tensor], List[str]]:
    """One full epoch passing.

    :param data: Data of the epoch.
    :type data: torch.utils.data.DataLoader
    :param module: Module to use.
    :type module: torch.nn.Module
    :param objective: Objective for the module.
    :type objective: callable|None
    :param optimizer: Optimizer for the module.
    :type optimizer: torch.optim.Optimizer | None
    :param use_y: Return the predictions and\
                  ground truth values? Defaults to False.
    :type use_y: bool
    :param grad_norm: Norm of the gradient for gradient clipping.
                      Defaults to 1. .
    :type grad_norm: int
    :param grad_norm_val: Max value for gradient clipping. If -1, then\
                          no clipping will happen. Defaults to -1. .
    :type grad_norm_val: float
    :return: Predicted and ground truth values\
             (if specified).
    :rtype: torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[str]
    """
    has_optimizer = optimizer is not None
    objective_output: Tensor = zeros(len(data))

    output_y_hat = []
    output_y = []
    f_names = []

    for i, example in enumerate(data):
        y_hat, y, f_names_tmp = module_forward_passing(example, module, use_y)
        f_names.extend(f_names_tmp)
        y = y[:, 1:]
        y_hat = y_hat.transpose(0, 1)
        if not use_y:  # inference
            y_hat = y_hat[:, 1:]

        try:
            if use_y:
                y_hat = y_hat[:, :y.size()[1], :]
            loss = objective(y_hat.contiguous().view(-1, y_hat.size()[-1]),
                             y.contiguous().view(-1))
            if has_optimizer:
                optimizer.zero_grad()
                if grad_norm_val > -1:
                    clip_grad_norm_(module.parameters(),
                                    max_norm=grad_norm_val,
                                    norm_type=grad_norm)
                loss.backward()
                optimizer.step()
                # scheduler.step()
                # plot_grad_flow(module.named_parameters())

            objective_output[i] = loss.item()
        except TypeError:
            pass
        try:
            output_y_hat.extend(y_hat.detach().cpu())
            output_y.extend(y.detach().cpu())
        except AttributeError:
            pass
        except TypeError:
            pass
    return objective_output, output_y_hat, output_y, f_names


def module_forward_passing(data: MutableSequence[Tensor],
                           module: Module,
                           use_y: bool) \
        -> Tuple[Tensor, Tensor, List[str]]:
    """One forward passing of the module.

    Implements one forward passing of the module `module`, using the provided\
    data `data`. Returns the output of the module and the ground truth values.

    :param data: Input and output values for current forward passing.
    :type data: list[torch.Tensor]
    :param module: Module.
    :type module: torch.nn.Module
    :param use_y: Use the ground truth as input to module?
    :type use_y: bool
    :return: Output of the module and target values.
    :rtype: torch.Tensor, torch.Tensor, list[str]
    """
    device = next(module.parameters()).device
    x, y, f_names = [i.to(device) if isinstance(i, Tensor)
                     else i for i in data]
    if use_y:
        return module(x,y), y, f_names
    else: 
        return module(x, None), y, f_names

# EOF
