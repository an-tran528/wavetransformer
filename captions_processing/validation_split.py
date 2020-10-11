#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from itertools import chain
import pickle
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from tools import csv_functions, captions_functions

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = []


def main():
    all_entries = csv_functions.read_csv_file(
        'clotho_captions_development.csv',
        'data')

    all_words = []

    for entry in all_entries:
        entry_words = [captions_functions.get_sentence_words(v, unique=True)
                       for k, v in entry.items() if not k.startswith('file')]
        all_words.extend(list(set(chain.from_iterable(entry_words))))

    counter = Counter(all_words)

    results = []
    max_min = 0
    files_to_use = []
    max_files = 50

    for entry in all_entries:
        captions = [v for k, v in entry.items() if not k.startswith('file')]
        min_freq = 1e6
        for caption in captions:
            min_freq = min(min_freq, *[counter.get(word) for word in
                           captions_functions.get_sentence_words(caption, unique=True)])
        max_min = max(max_min, min_freq)
        results.append({'file': entry.get('file_name'), 'min_freq': min_freq})
        if 10 < min_freq < 20:
            files_to_use.append(entry.get('file_name'))

    print(f'Max minimum freq is {max_min}')
    print(f'Amount of files that I can use is {len(files_to_use)}')
    # plt.hist([k['min_freq'] for k in results],
    #          bins=max_min,
    #          histtype='stepfilled')
    # plt.grid()
    # plt.show()

    x = np.arange(len(files_to_use))
    final_files = [files_to_use[i] for i in np.random.permutation(x)[:max_files]]
    [print(f'File {i+1:02d} is {f}') for i, f in enumerate(final_files)]

    p = Path('validation_file_names.pickle')

    print('Saving list of validation files...', end=' ')
    with p.open('wb') as f:
        pickle.dump(final_files, f)
    print('done.')


if __name__ == '__main__':
    main()

# EOF
