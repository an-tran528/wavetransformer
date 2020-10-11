#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import stdout
from pathlib import Path
from pprint import PrettyPrinter
from argparse import ArgumentParser

from loguru import logger
from _io import TextIOWrapper
from matplotlib import pyplot as plt

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_pretty_printer',
           'init_loggers']


def _rotation_logger(x: str,
                     y: TextIOWrapper) \
        -> bool:
    """Callable to determine the rotation of files in logger.

    :param x: Str to be logged.
    :type x: loguru._handler.StrRecord
    :param y: File used for logging.
    :type y: _io.TextIOWrapper
    :return: Shall we switch to a new file?
    :rtype: bool
    """
    return 'Captions start' in x


def get_pretty_printer():
    """Gets the pprint.

    :return: Pretty printer.
    :rtype: pprint.PrettyPrinter
    """
    return PrettyPrinter(indent=4, width=100)


def init_loggers(job_id, verbose, settings):
    """Initializes the logging process.

    :param job_id: Unique job identifier.
    :type job_id: str
    :param verbose: Be verbose?
    :type verbose: bool
    :param settings: Settings to use.
    :type settings: dict
    """
    logger.remove()

    for indent in range(3):
        log_string = '{level} | [{time:HH:mm:ss}] {name} -- {message}'.rjust(indent*2)
        logger.add(
            stdout,
            format=log_string,
            level='INFO',
            filter=lambda record, i=indent:
            record['extra']['indent'] == i and not record['extra']['is_caption'])

    logging_path = Path(settings['root_dirs']['outputs'],
                        settings['logging']['logger_dir'])

    log_file_main = f'{settings["logging"]["caption_logger_file"]}'

    logging_file = logging_path.joinpath(log_file_main)

    logger.add(str(logging_file), format='{message}', level='INFO',
               filter=lambda record: record['extra']['is_caption'],
               rotation=_rotation_logger)

    logging_path.mkdir(parents=True, exist_ok=True)

    if not verbose:
        logger.disable('__main__')


def print_results(file_name_source: str,
                  file_name_output: str,
                  dpi: int,
                  output_dir: str) \
        -> None:
    """Prints the results in the specified file.

    :param file_name_source: File with values.
    :type file_name_source: str
    :param file_name_output: Plotting of the values.
    :type file_name_output: str
    :param dpi: DPI of the resulting image.
    :type dpi: int
    :param output_dir: Directory for saving the file in outputs directory.
    :type output_dir: str
    """
    with Path(file_name_source).open('r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if '-- Epoch:' in line]

    tr_loss, va_loss, epochs = [], [], []

    for line in lines:
        line_ = line.split('-- Epoch:')[1].strip().split('--')
        epoch = int(line_[0].strip())
        tr_va = line_[1].strip().split('|')[0].split(':')[1].strip()
        tr, va = [float(i.strip()) for i in tr_va.split('/')]

        tr_loss.append(tr)
        va_loss.append(va)
        epochs.append(epoch)

    o_dir = Path('outputs', output_dir)

    if not o_dir.exists():
        o_dir.mkdir(parents=True, exist_ok=True)

    o_f_path = Path(file_name_output).name

    plt.plot(epochs, tr_loss, epochs, va_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(str(o_dir.joinpath(o_f_path)), dpi=dpi)


def main():
    args_parser = ArgumentParser()
    args = [
        [['--input-file', '-i'],
         {'type': str, 'required': True,
          'help': 'The file with the values'}],
        [['--output-file', '-o'],
         {'type': str, 'required': True,
          'help': 'The file with the plot'}],
        [['--output-dir', '-d'],
         {'type': str, 'default': 'images',
          'help': 'Directory for saving the file in outputs directory.'}],
        [['--dpi'],
         {'type': int, 'default': 900,
          'help': 'DPI of the resulting image'}]]

    [args_parser.add_argument(*i[0], **i[1]) for i in args]

    args_o = args_parser.parse_args()

    print_results(file_name_source=args_o.input_file,
                  file_name_output=args_o.output_file,
                  dpi=args_o.dpi,
                  output_dir=args_o.output_dir)


if __name__ == '__main__':
    main()

# EOF
