#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple, Optional, Union
from pathlib import Path
from itertools import groupby

from torch.utils.data import Dataset
from numpy import load as np_load, ndarray

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['ClothoDataset']


def _get_audio_file_name(s: Path) \
        -> str:
    """Returns the original audio file name from clotho file.

    :param s: Clotho file name.
    :type s: pathlib.Path
    :return: Original audio file name.
    :rtype: str
    """
    return s.stem.split('clotho_file_')[-1].split('.wav')[0]


class ClothoDataset(Dataset):

    def __init__(self,
                 data_dir: Path,
                 split: str,
                 input_field_name: str,
                 output_field_name: str,
                 load_into_memory: bool,
                 multiple_captions_mode: Optional[bool] = False,
                 validation_files: Optional[Union[List[Path], None]] = None) \
            -> None:
        """Initialization of a Clotho dataset object.

        :param data_dir: Data directory with Clotho dataset files.
        :type data_dir: pathlib.Path
        :param split: The split to use (`development`, `validation`)
        :type split: str
        :param input_field_name: Field name for the input values
        :type input_field_name: str
        :param output_field_name: Field name for the output (target) values.
        :type output_field_name: str
        :param load_into_memory: Load the dataset into memory?
        :type load_into_memory: bool
        :param multiple_captions_mode: Use all captions of the same sound\
                                       in the same batch? Defaults to False.
        :type multiple_captions_mode: bool, optional
        :param validation_files: Files ot be used as validation set (applicable only
                                when development split is used), defaults to None.
        :type validation_files: list[pathlib.Path] | None, optional
        """
        super(ClothoDataset, self).__init__()
        the_dir = data_dir.joinpath(split)

        self.multiple_captions_mode = multiple_captions_mode

        self.examples: List[Path] = sorted([
            i for i in the_dir.iterdir() if i.suffix == '.npy'])

        if split.lower() in ['development', 'validation']:
            validation_stems = [] if validation_files is None \
                else [v_i.stem for v_i in validation_files]

            self.examples = [s_i for s_i in self.examples
                             if (_get_audio_file_name(s_i) in validation_stems) or
                             (split.lower() == 'development')]
        if self.multiple_captions_mode:
            self.examples: List[List[Path]] = [
                list(v) for _, v in groupby(
                    self.examples,
                    _get_audio_file_name)]

        self.input_name = input_field_name
        self.output_name = output_field_name
        self.load_into_memory = load_into_memory

        if load_into_memory:
            if self.multiple_captions_mode:
                tmp = []
                for tmp_l in self.examples:
                    tmp.append([np_load(str(f), allow_pickle=True)
                               for f in tmp_l])
                self.examples: List[List[ndarray]] = tmp
            else:
                self.examples: List[ndarray] = [
                    np_load(str(f), allow_pickle=True)
                    for f in self.examples]

    def __len__(self) \
            -> int:
        """Gets the amount of examples in the dataset.

        :return: Amount of examples in the dataset.
        :rtype: int
        """
        return len(self.examples)

    def __getitem__(self,
                    item: int) \
            -> Tuple[List[ndarray], List[ndarray], List[Path]]:
        """Gets an example from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Input and output values, and the Path of the file.
        :rtype: list[numpy.ndarray], list[numpy.ndarray], list[Path]
        """
        ex = self.examples[item] if self.multiple_captions_mode \
            else [self.examples[item]]

        if not self.load_into_memory:
            ex = [np_load(str(i), allow_pickle=True) for i in ex]

        x, y, file_names = [], [], []
        for ex_i in ex:
            x.append(ex_i[self.input_name])
            y.append(ex_i[self.output_name])
            file_names.append(Path(ex_i.file_name[0]))
        return x, y, file_names

# EOF
