#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from processes import method
from tools.file_io import load_yaml_file
from tools.printing import init_loggers
from tools.argument_parsing import get_argument_parser

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['main']


def main():
    args = get_argument_parser().parse_args()

    file_dir = args.file_dir
    config_file = args.config_file
    file_ext = args.file_ext
    verbose = args.verbose
    job_id = args.job_id

    settings = load_yaml_file(Path(
        file_dir, f'{config_file}.{file_ext}'))

    init_loggers(verbose=verbose,
                 settings=settings['dirs_and_files'],
                 job_id=job_id)

    method.method(settings, job_id)


if __name__ == '__main__':
    main()

# EOF
