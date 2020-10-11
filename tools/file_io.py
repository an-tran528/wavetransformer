#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import pickle
import yaml
import numpy as np
from librosa import load

from tools import yaml_loader

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = [
    'dump_pickle_file', 'load_pickle_file', 'read_txt_file',
    'load_audio_file', 'dump_numpy_object', 'load_numpy_object',
    'load_yaml_file', 'load_settings_file'
]


def dump_numpy_object(np_obj, file_name, ext='.npy', replace_ext=True):
    """Dumps a numpy object to HDD.

    :param np_obj: Numpy object.
    :type np_obj: numpy.ndarray
    :param file_name: File name to be used.
    :type file_name: pathlib.Path
    :param ext: Extension for the dumped object.
    :type ext: str
    :param replace_ext: Replace extension?
    :type replace_ext: bool
    """
    f_name = file_name.with_suffix(ext) if replace_ext else file_name
    np.save(str(f_name), np_obj)


def dump_pickle_file(obj, file_name, protocol=2):
    """Dumps an object to pickle file.

    :param obj: Object to dump.
    :type obj: object | list | dict | numpy.ndarray
    :param file_name: Resulting file name.
    :type file_name: pathlib.Path
    :param protocol: Protocol to be used.
    :type protocol: int
    """
    with file_name.open('wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def load_audio_file(audio_file, sr, mono, offset=0.0, duration=None):
    """Loads the data of an audio file.

    :param audio_file: Path of the audio file.
    :type audio_file: pathlib.Path
    :param sr: Sampling frequency to be used.
    :type sr: int
    :param mono: Turn to mono?
    :type mono: bool
    :param offset: Offset to be used (in seconds).
    :type offset: float
    :param duration: Duration of signal to load (in seconds).
    :type duration: float|None
    :return: Audio data.
    :rtype: numpy.ndarray
    """
    return load(path=str(audio_file), sr=sr, mono=mono,
                offset=offset, duration=duration)[0]


def load_numpy_object(f_name):
    """Loads and returns a numpy object.

    :param f_name: Path of the object.
    :type f_name: pathlib.Path
    :return: Numpy object.
    :rtype: numpy.ndarray
    """
    return np.load(str(f_name), allow_pickle=True)


def load_pickle_file(file_name, encoding='latin1'):
    """Loads a pickle file.

    :param file_name: File name (extension included).
    :type file_name: pathlib.Path
    :param encoding: Encoding of the file.
    :type encoding: str
    :return: Loaded object.
    :rtype: object | list | dict | numpy.ndarray
    """
    with file_name.open('rb') as f:
        return pickle.load(f, encoding=encoding)


def load_settings_file(file_name, settings_dir=pathlib.Path('settings')):
    """Reads and returns the contents of a YAML settings file.

    :param file_name: Name of the settings file.
    :type file_name: pathlib.Path
    :param settings_dir: Directory with the settings files.
    :type settings_dir: pathlib.Path
    :return: Contents of the YAML settings file.
    :rtype: dict
    """
    settings_file_path = settings_dir.joinpath(file_name.with_suffix('.yaml'))
    return load_yaml_file(settings_file_path)


def load_yaml_file(file_path):
    """Reads and returns the contents of a YAML file.

    :param file_path: Path to the YAML file.
    :type file_path: pathlib.Path
    :return: Contents of the YAML file.
    :rtype: dict
    """
    with file_path.open('r') as f:
        return yaml.load(f, Loader=yaml_loader.YAMLLoader)


def read_txt_file(file_name):
    """Reads a text (.txt) file and returns the contents.

    :param file_name: File name of the txt file.
    :type file_name: pathlib.Path
    :return: Contents of the file.
    :rtype: list[str]
    """
    with file_name.open() as f:
        return f.readlines()

# EOF
