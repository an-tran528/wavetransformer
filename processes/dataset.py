#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableMapping, Any
from datetime import datetime
from pathlib import Path
from functools import partial
from itertools import chain

import numpy as np
from loguru import logger

from tools.printing import init_loggers
from tools.argument_parsing import get_argument_parser
from tools.dataset_creation import get_annotations_files, \
    get_amount_of_file_in_dir, check_data_for_split, \
    create_split_data, create_lists_and_frequencies
from tools.file_io import load_settings_file, load_yaml_file, \
    load_pickle_file, load_numpy_object, dump_numpy_object, load_audio_file
from tools.features_log_mel_bands import feature_extraction

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['create_dataset', 'extract_features']


def create_dataset(settings_dataset: MutableMapping[str, Any],
                   settings_dirs_and_files: MutableMapping[str, Any]) \
        -> None:
    """Creates the dataset.

    Gets the dictionary with the settings and creates
    the files of the dataset.

    :param settings_dataset: Settings to be used for dataset\
                             creation.
    :type settings_dataset: dict
    :param settings_dirs_and_files: Settings to be used for\
                                    handling directories and\
                                    files.
    :type settings_dirs_and_files: dict
    """
    # Get logger
    inner_logger = logger.bind(
        indent=2, is_caption=False)

    # Get root dir
    dir_root = Path(settings_dirs_and_files[
                        'root_dirs']['data'])

    # Read the annotation files
    inner_logger.info('Reading annotations files')
    csv_dev, csv_eva = get_annotations_files(
        settings_ann=settings_dataset['annotations'],
        dir_ann=dir_root.joinpath(
            settings_dirs_and_files['dataset'][
                'annotations_dir']))
    inner_logger.info('Done')

    # Get all captions
    inner_logger.info('Getting the captions')
    captions_development = [
        csv_field.get(
            settings_dataset['annotations'][
                'captions_fields_prefix'].format(c_ind))
        for csv_field in csv_dev
        for c_ind in range(1, 6)]
    inner_logger.info('Done')

    # Create lists of indices and frequencies for words and\
    # characters.
    inner_logger.info('Creating and saving words and chars '
                      'lists and frequencies')
    words_list, chars_list = create_lists_and_frequencies(
        captions=captions_development, dir_root=dir_root,
        settings_ann=settings_dataset['annotations'],
        settings_cntr=settings_dirs_and_files['dataset'])
    inner_logger.info('Done')

    # Aux partial function for convenience.
    split_func = partial(
        create_split_data,
        words_list=words_list, chars_list=chars_list,
        settings_ann=settings_dataset['annotations'],
        settings_audio=settings_dataset['audio'],
        settings_output=settings_dirs_and_files['dataset'])

    settings_audio_dirs = settings_dirs_and_files[
        'dataset']['audio_dirs']

    # For each data split (i.e. development and evaluation)
    for split_data in [(csv_dev, 'development'),
                       (csv_eva, 'evaluation')]:

        # Get helper variables.
        split_name = split_data[1]
        split_csv = split_data[0]

        dir_split = dir_root.joinpath(
            settings_audio_dirs['output'],
            settings_audio_dirs[f'{split_name}'])

        dir_downloaded_audio = dir_root.joinpath(
            settings_audio_dirs['downloaded'],
            settings_audio_dirs[f'{split_name}'])

        # Create the data for the split.
        inner_logger.info(f'Creating the {split_name} '
                          f'split data')
        split_func(split_csv, dir_split,
                   dir_downloaded_audio)
        inner_logger.info('Done')

        # Count and print the amount of initial and resulting\
        # files.
        nb_files_audio = get_amount_of_file_in_dir(
            dir_downloaded_audio)
        nb_files_data = get_amount_of_file_in_dir(dir_split)

        inner_logger.info(f'Amount of {split_name} '
                          f'audio files: {nb_files_audio}')
        inner_logger.info(f'Amount of {split_name} '
                          f'data files: {nb_files_data}')
        inner_logger.info(f'Amount of {split_name} data '
                          f'files per audio: '
                          f'{nb_files_data / nb_files_audio}')

        if settings_dataset['workflow']['validate_dataset']:
            # Check the created lists of indices for words and characters.
            inner_logger.info(f'Checking the {split_name} split')
            check_data_for_split(
                dir_audio=dir_downloaded_audio,
                dir_data=Path(settings_audio_dirs['output'],
                              settings_audio_dirs[f'{split_name}']),
                dir_root=dir_root, csv_split=split_csv,
                settings_ann=settings_dataset['annotations'],
                settings_audio=settings_dataset['audio'],
                settings_cntr=settings_dirs_and_files['dataset'])
            inner_logger.info('Done')
        else:
            inner_logger.info(f'Skipping validation of {split_name} split')


def extract_features(root_dir: str,
                     settings_data: MutableMapping[str, Any],
                     settings_features: MutableMapping[str, Any]) \
        -> None:
    """Extracts features from the audio data of Clotho.

    :param root_dir: Root dir for the data.
    :type root_dir: str
    :param settings_data: Settings for creating data files.
    :type settings_data: dict[str, T]
    :param settings_features: Settings for feature extraction.
    :type settings_features: dict[str, T]
    """
    # Get the root directory.
    dir_root = Path(root_dir)

    # Get the directories of files.
    dir_output = dir_root.joinpath(settings_data['audio_dirs']['output'])

    dir_dev = dir_output.joinpath(
        settings_data['audio_dirs']['development'])
    dir_eva = dir_output.joinpath(
        settings_data['audio_dirs']['evaluation'])

    # Get the directories for output.
    dir_output_dev = dir_root.joinpath(
        settings_data['features_dirs']['output'],
        settings_data['features_dirs']['development'])
    dir_output_eva = dir_root.joinpath(
        settings_data['features_dirs']['output'],
        settings_data['features_dirs']['evaluation'])

    # Create the directories.
    dir_output_dev.mkdir(parents=True, exist_ok=True)
    dir_output_eva.mkdir(parents=True, exist_ok=True)

    # Apply the function to each file and save the result.
    for data_file_name in filter(
            lambda _x: _x.suffix == '.npy',
            chain(dir_dev.iterdir(), dir_eva.iterdir())):

        # Load the data file.
        data_file = load_numpy_object(data_file_name)

        # Extract the features.
        features = feature_extraction(
            data_file['audio_data'].item(),
            **settings_features['process'])

        # Populate the recarray data and dtypes.
        array_data = (data_file['file_name'].item(),)
        dtypes = [('file_name', data_file['file_name'].dtype)]

        # Check if we keeping the raw audio data.
        if settings_features['keep_raw_audio_data']:
            # And add them to the recarray data and dtypes.
            array_data += (data_file['audio_data'].item(),)
            dtypes.append(('audio_data', data_file['audio_data'].dtype))

        # Add the rest to the recarray.
        array_data += (
            features,
            data_file['caption'].item(),
            data_file['caption_ind'].item(),
            data_file['words_ind'].item(),
            data_file['chars_ind'].item())
        dtypes.extend([
            ('features', np.dtype(object)),
            ('caption', data_file['caption'].dtype),
            ('caption_ind', data_file['caption_ind'].dtype),
            ('words_ind', data_file['words_ind'].dtype),
            ('chars_ind', data_file['chars_ind'].dtype)
        ])

        # Make the recarray
        np_rec_array = np.rec.array([array_data], dtype=dtypes)

        # Make the path for serializing the recarray.
        parent_path = dir_output_dev \
            if data_file_name.parent.name == settings_data['audio_dirs']['development'] \
            else dir_output_eva

        file_path = parent_path.joinpath(data_file_name.name)

        # Dump it.
        dump_numpy_object(np_rec_array, file_path)


def extract_features_test(root_dir: str,
                          settings_data: MutableMapping[str, Any],
                          settings_features: MutableMapping[str, Any],
                          settings_audio: MutableMapping[str, Any]) \
        -> None:
    """Extracts test features from the audio data of Clotho.

    :param root_dir: Root dir for the data.
    :type root_dir: str
    :param settings_data: Settings for creating data files.
    :type settings_data: dict[str, T]
    :param settings_features: Settings for feature extraction.
    :type settings_features: dict[str, T]
    :param settings_audio: Settings for the audio.
    :type settings_audio: dict
    """
    # Get the root directory.
    dir_root = Path(root_dir)

    # Get the directories of files.
    dir_test = dir_root.joinpath(
        settings_data['audio_dirs']['downloaded'],
        settings_data['audio_dirs']['test'])

    audio_exists = False
    if dir_test.exists() and len(list(dir_test.iterdir())) != 0:
        audio_exists = True
    if not audio_exists:
        raise AttributeError('Testing workflow selected, but could not find the test set audio files. '
                             'Please download the test set audio before making test predictions.')

    # Get the directories for output.
    dir_output_test = dir_root.joinpath(
        settings_data['features_dirs']['output'],
        settings_data['features_dirs']['test'])

    words_list = load_pickle_file(
        dir_root.joinpath(
            settings_data['pickle_files_dir'],
            settings_data['files']['words_list_file_name']))

    # Create the directories.
    dir_output_test.mkdir(parents=True, exist_ok=True)

    # Apply the function to each file and save the result.
    for data_file_name in filter(
            lambda _x: _x.is_file(),
            dir_test.iterdir()):
        # Load the audio
        audio = load_audio_file(
            audio_file=str(data_file_name),
            sr=int(settings_audio['sr']),
            mono=settings_audio['to_mono'])

        # Extract the features.
        features = feature_extraction(
            audio,
            **settings_features['process'])

        # Populate the recarray data and dtypes.
        array_data = (data_file_name.name,)
        dtypes = [('file_name', f'U{len(data_file_name.name)}')]

        # Check if we keeping the raw audio data.
        if settings_features['keep_raw_audio_data']:
            # And add them to the recarray data and dtypes.
            array_data += (audio,)
            dtypes.append(('audio_data', audio.dtype))

        # Add the rest to the recarray.
        # Word indices are required for the dataloader to work
        array_data += (features,
                       np.array([words_list.index('<sos>'), words_list.index('<eos>')]))
        dtypes.extend([
            ('features', np.dtype(object)),
            ('words_ind', np.dtype(object))])

        # Make the recarray
        np_rec_array = np.rec.array([array_data], dtype=dtypes)

        # Make the path for serializing the recarray.
        parent_path = dir_output_test

        file_template = settings_data['files']['np_file_name_template'].replace('_{caption_index}', '')
        file_path = parent_path.joinpath(file_template.format(audio_file_name=data_file_name.name))

        # Dump it.
        dump_numpy_object(np_rec_array, file_path)


def main():
    args = get_argument_parser().parse_args()

    file_dir = args.file_dir
    config_file = args.config_file
    file_ext = args.file_ext
    verbose = args.verbose

    # Load settings file.
    settings = load_yaml_file(Path(
        file_dir, f'{config_file}.{file_ext}'))

    init_loggers(verbose=verbose,
                 settings=settings['dirs_and_files'])

    logger_main = logger.bind(is_caption=False, indent=0)
    logger_sec = logger.bind(is_caption=False, indent=1)

    logger_main.info(datetime.now().strftime('%Y-%m-%d %H:%M'))

    logger_main.info('Doing only dataset creation')

    # Create the dataset.
    logger_main.info('Starting Clotho dataset creation')

    logger_sec.info('Creating examples')
    create_dataset(
        settings_dataset=settings['dataset_creation_settings'],
        settings_dirs_and_files=settings['dirs_and_files'])
    logger_sec.info('Examples created')

    logger_sec.info('Extracting features')
    extract_features(
        root_dir=settings['dirs_and_files']['root_dirs']['data'],
        settings_data=settings['dirs_and_files']['dataset'],
        settings_features=settings['feature_extraction_settings'])
    logger_sec.info('Features extracted')

    logger_main.info('Dataset created')


if __name__ == '__main__':
    main()

# EOF
