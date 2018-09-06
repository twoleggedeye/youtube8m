import glob
import os
import pickle

import numpy as np

from youtube8m.video_level_nn_models.chunked_dataset import ChunkedMultiModalDataset, FakeModality, SparseModality
from youtube8m.video_level_nn_models.defaults import YOUTUBE8M_LABELS_N


class PaddedPickleModality(object):
    def __init__(self, filename, norm_ord=2, norm_axis=2):
        self.filename = filename
        self._norm_ord = norm_ord
        self._norm_axis = norm_axis

    def load(self, dirname):
        with open(os.path.join(dirname, self.filename), 'rb') as fin:
            return pickle.load(fin)

    def convert_batch(self, batch):
        max_batch_len = max(s.shape[0] for s in batch)
        result = np.zeros((len(batch), max_batch_len, batch[0].shape[1]), dtype='float32')
        for i, sample in enumerate(batch):
            result[i, :len(sample), :] = sample / 64. - 2
        return result


def make_frame_level_video_audio_unlabeled_dataset(in_glob, batch_size=64):
    return ChunkedMultiModalDataset(glob.glob(os.path.join(in_glob, 'chunk*')),
                                    [PaddedPickleModality('rgb'),
                                     PaddedPickleModality('audio'),
                                     FakeModality()],
                                    2,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    drop_small_batch=False)


def make_frame_level_video_audio_labels_dataset(in_glob, batch_size=64, labels_n=YOUTUBE8M_LABELS_N,
                                                shuffle=True, drop_small_batch=False):
    return ChunkedMultiModalDataset(glob.glob(os.path.join(in_glob, 'chunk*')),
                                    [PaddedPickleModality('rgb'),
                                     PaddedPickleModality('audio'),
                                     SparseModality('labels', labels_n)],
                                    2,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    drop_small_batch=drop_small_batch)
