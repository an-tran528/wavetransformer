#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableSequence, List, Dict, Tuple
from pathlib import Path
from itertools import chain
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

from tools import csv_functions, captions_functions

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = []


def get_all_captions_from_files(files: MutableSequence[Path],
                                data_dir: Path) \
        -> List[Dict[str, str]]:
    all_entries = chain.from_iterable([
        csv_functions.read_csv_file(a_f, data_dir)
        for a_f in files])

    return [{k: v for k, v in i.items() if not k.startswith('file')}
            for i in all_entries]


def get_hist_of_words(files: MutableSequence[Path],
                      data_dir: Path) \
        -> Tuple[List[str], List[int]]:
    all_captions = get_all_captions_from_files(
        files=files, data_dir=data_dir)

    c = Counter(chain.from_iterable(
        [captions_functions.get_sentence_words(v, unique=False)
         for i in all_captions for v in i.values()]))

    return list(c.keys()), list(c.values())


def plot_hist_of_words():
    dev_captions = Path('clotho_captions_development.csv')
    data_dir = Path('data')
    values = np.array(get_hist_of_words([dev_captions], data_dir)[1])
    values = values.max()/values
    values = values/values.max()
    values_clamped_0_01 = values.clip(0.01, values.max())
    values_clamped_0_001 = values.clip(0.001, values.max())
    plt.hist(values_clamped_0_001, bins=1000, color='green')
    plt.hist(values_clamped_0_01, bins=1000, color='blue')
    plt.hist(values, bins=1000, color='orange')
    plt.legend(
        ['No clamping', 'Clamping at 0.01', 'Clamping at 0.001']
    )
    plt.show()
    print('Min values:')
    print(f'\tNo clamping      : {values.min()}')
    print(f'\tClamping at 0.01 : {values_clamped_0_01.min()}')
    print(f'\tClamping at 0.001: {values_clamped_0_001.min()}')


def plot_tf_prob(mul_factor: int,
                 every_five: bool):
    nb_audio_files = len(list(Path('data', 'development').iterdir()))
    nb_examples = nb_audio_files * 5

    gamma_factor = 10.0/mul_factor

    if every_five:
        epoch_mul = nb_audio_files
    else:
        epoch_mul = nb_examples

    p = np.arange(1000 * epoch_mul/16.0)/(epoch_mul/16.0)
    d = np.exp((-gamma_factor * p))

    plt.figure()
    plt.plot(np.minimum(0.9, 1 - np.minimum(0.95, (2. / (1. + d)) - 1.)))

    x_100 = (np.ones(100) * (epoch_mul/16.0) * 100 * 1, np.arange(100)/100)
    x_200 = (np.ones(100) * (epoch_mul/16.0) * 100 * 2, np.arange(100)/100)
    x_300 = (np.ones(100) * (epoch_mul/16.0) * 100 * 3, np.arange(100)/100)
    x_400 = (np.ones(100) * (epoch_mul/16.0) * 100 * 4, np.arange(100)/100)

    plt.plot(*x_100)
    plt.plot(*x_200)
    plt.plot(*x_300)
    plt.plot(*x_400)

    plt.show()


def captions_length():
    all_captions = get_all_captions_from_files(
        files=[Path('clotho_captions_development.csv')],
        data_dir=Path('data'))
    lengths = [len(v.split()) for c in all_captions for v in c.values()]
    print(f'Max length is {max(lengths)}')
    plt.figure()
    plt.hist(lengths, bins=22)
    plt.show()


def get_max_captions_length():
    all_captions = get_all_captions_from_files(
        files=[Path('clotho_captions_development.csv')],
        data_dir=Path('data'))
    lengths = [len(v.split()) for c in all_captions for v in c.values()]

    return max(lengths)


def distribution_per_output_t_step():
    all_captions = [v.split() for k in get_all_captions_from_files(
        files=[Path('clotho_captions_development.csv')],
        data_dir=Path('data')) for v in k.values()]

    counter = Counter(chain.from_iterable(all_captions))

    labels = list(counter.keys())

    max_length = get_max_captions_length()
    plt.figure()

    for i in range(max_length):
        print(f'Processing {i}')
        tmp_c = Counter(q[i] for q in all_captions if i < len(q))
        plt.subplot(6, 4, i+1)
        values = [tmp_c[k] if k in tmp_c.keys() else 0 for k in labels]
        values = np.array(values)/max(values)
        plt.bar(range(len(labels)), values)
        plt.title(f'Time step {i+1}')
        plt.ylim(0, 1)
    plt.tight_layout(pad=1.0)
    plt.savefig(f'/Users/konstantinosdrosos/PycharmProjects/audio-captioning-lm/tmp.png',
                dpi=900, quality=100, format='png', transparent=True)
    print('Exited loop')


def main():
    # plot_hist_of_words()
    # plot_tf_prob(mul_factor=1000, every_five=True)
    # captions_length()
    distribution_per_output_t_step()
    pass


if __name__ == '__main__':
    main()

# EOF
