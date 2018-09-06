import collections
import glob
import itertools
import os

import numpy as np
import pandas as pd
import scipy.sparse
import torch

from youtube8m.utils.io import ensure_dir_exists, sort_files_natural, load_pickle
from youtube8m.video_level_nn_models.defaults import YOUTUBE8M_LABELS_N


def get_chunk_sizes(files, mmap_mode='r'):
    result = []
    for i, fname in enumerate(files):
        cur_size = np.load(fname, mmap_mode=mmap_mode).shape[0]
        result.append(cur_size)
    return result


class DenseModalityAnyType(object):
    def __init__(self, filename):
        self.filename = filename

    def load(self, dirname):
        return np.load(os.path.join(dirname, self.filename), mmap_mode='r')

    def convert_batch(self, batch):
        return np.array(batch)


class DenseModality(DenseModalityAnyType):
    def convert_batch(self, batch):
        return super().convert_batch(batch).astype('float32')


class SparseModality(object):
    def __init__(self, filename, size):
        self.filename = filename
        self.size = size

    def load(self, dirname):
        return np.load(os.path.join(dirname, self.filename))

    def convert_batch(self, batch):
        result = np.zeros((len(batch), self.size), dtype='float32')
        for row in range(len(batch)):
            for col in batch[row]:
                result[row, col] = 1
        return result


class SparseModalityWithMetaTags(object):
    def __init__(self, filename, size, vocabulary_path):
        self.filename = filename
        self.size = size
        vocab = pd.read_csv(vocabulary_path, index_col=['Index'])
        all_tags = sorted(set(itertools.chain(vocab['Vertical1'].dropna(),
                                              vocab['Vertical2'].dropna(),
                                              vocab['Vertical3'].dropna())))
        tag2i = {t: i for i, t in enumerate(all_tags)}
        self._cat_to_tag_ids = collections.defaultdict(set)
        for i in vocab.index:
            for tag in vocab.iloc[i][['Vertical1', 'Vertical2', 'Vertical3']].dropna():
                self._cat_to_tag_ids[i].add(tag2i[tag])
        self._tags_n = len(tag2i)

    def load(self, dirname):
        return np.load(os.path.join(dirname, self.filename))

    def convert_batch(self, batch):
        result = np.zeros((len(batch), self.size + self._tags_n), dtype='float32')
        for row in range(len(batch)):
            for col in batch[row]:
                result[row, col] = 1
                for tag in self._cat_to_tag_ids[col]:
                    result[row, self.size + tag] = 1
        return result


class FakeArray(object):
    def __getitem__(self, *args, **kwargs):
        return np.zeros(0)


class FakeModality(object):
    def load(self, _):
        return FakeArray()

    def convert_batch(self, batch):
        return batch


def any_to_tensor(x):
    if isinstance(x, np.ndarray) and x.dtype.type == np.str_:
        return x
    return torch.from_numpy(x)


class ChunkedMultiModalDataset(object):
    def __init__(self, chunk_dirs, modalities, in_modalities_n, batch_size=64, shuffle=True, drop_small_batch=False,
                 data_to_torch=any_to_tensor):
        self._chunk_dirs = sort_files_natural(chunk_dirs)
        self._modalities = modalities
        self._in_modalities_n = in_modalities_n
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_small_batch = drop_small_batch
        self._data_to_torch = data_to_torch

    def __iter__(self):
        dirs_in_order = list(self._chunk_dirs)
        if self._shuffle:
            np.random.shuffle(dirs_in_order)

        for chunk_dir in dirs_in_order:
            chunk_mod_data = [modality.load(chunk_dir) for modality in self._modalities]
            indices = list(range(chunk_mod_data[0].shape[0]))
            if self._shuffle:
                np.random.shuffle(indices)

            for batch_start in range(0, len(indices), self._batch_size):
                cur_batch_size = min(len(indices) - batch_start, self._batch_size)
                if cur_batch_size < self._batch_size and self._drop_small_batch:
                    continue

                batch_indices = indices[batch_start:batch_start+cur_batch_size]
                batch_modalities = [self._data_to_torch(modality.convert_batch(mod_data[batch_indices]))
                                    for modality, mod_data in zip(self._modalities, chunk_mod_data)]
                input_batch = batch_modalities[:self._in_modalities_n]
                output_batch = batch_modalities[self._in_modalities_n:]

                if len(output_batch) == 1:
                    output_batch = output_batch[0]

                yield input_batch, output_batch


class DatasetWithSoftLabels(object):
    def __init__(self, base_dataset, soft_idx_path, soft_scores_path, soft_weight=1.0):
        self._base_dataset = base_dataset
        self._soft_idx = load_pickle(soft_idx_path)
        self._soft_scores_path = soft_scores_path
        self._soft_scores = None
        self._soft_weight = soft_weight

    def __iter__(self):
        soft_idx = dict(self._soft_idx)
        if self._soft_scores is None:
            self._soft_scores = scipy.sparse.load_npz(self._soft_scores_path)
        soft_scores = self._soft_scores
        for (batch_ids, *features), batch_hard_labels in self._base_dataset:
            soft_ids = [soft_idx[i] for i in batch_ids]
            batch_soft_labels = torch.from_numpy(soft_scores[soft_ids].toarray().astype('float32'))
            mixed_labels = self._soft_weight * batch_soft_labels + (1 - self._soft_weight) * batch_hard_labels

            yield features, mixed_labels


def make_video_audio_unlabeled_dataset(in_glob, batch_size=64):
    return ChunkedMultiModalDataset(glob.glob(os.path.join(in_glob, 'chunk*')),
                                    [DenseModality('rgb'),
                                     DenseModality('audio'),
                                     FakeModality()],
                                    2,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    drop_small_batch=False)


def make_video_audio_labels_dataset(in_glob, batch_size=64, labels_n=YOUTUBE8M_LABELS_N, shuffle=True, drop_small_batch=False):
    return ChunkedMultiModalDataset(glob.glob(os.path.join(in_glob, 'chunk*')),
                                    [DenseModality('rgb'),
                                     DenseModality('audio'),
                                     SparseModality('labels', labels_n)],
                                    2,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    drop_small_batch=drop_small_batch)


def make_video_audio_soft_labels_dataset(in_glob, batch_size=64, labels_n=YOUTUBE8M_LABELS_N, shuffle=True,
                                         drop_small_batch=False,
                                         soft_idx='/Vol1/dbstore/datasets/multimodal/youtube/numpy/video_level_lvl2_index/idx.pkl',
                                         soft_scores='/Vol1/dbstore/datasets/multimodal/youtube/numpy/video_level_lvl2_index/idx.pkl.matr.npz',
                                         soft_weight=1.0):
    base_dataset = ChunkedMultiModalDataset(glob.glob(os.path.join(in_glob, 'chunk*')),
                                            [DenseModalityAnyType('video_ids'),
                                             DenseModality('rgb'),
                                             DenseModality('audio'),
                                             SparseModality('labels', labels_n)],
                                            3,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            drop_small_batch=drop_small_batch)
    return DatasetWithSoftLabels(base_dataset, soft_idx, soft_scores, soft_weight=soft_weight)


def make_video_audio_labels_metatags_dataset(in_glob, batch_size=64, labels_n=YOUTUBE8M_LABELS_N, shuffle=True,
                                             drop_small_batch=False,
                                             vocabulary='/home/rsuvorov/projects/youtube8m/vocabulary.csv'):
    return ChunkedMultiModalDataset(glob.glob(os.path.join(in_glob, 'chunk*')),
                                    [DenseModality('rgb'),
                                     DenseModality('audio'),
                                     SparseModalityWithMetaTags('labels', labels_n, vocabulary)],
                                    2,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    drop_small_batch=drop_small_batch)


make_video_audio_unlabeled_metatags_dataset = make_video_audio_unlabeled_dataset


class ChunkedWriter(object):
    def __init__(self, out_dir, min_chunk_size=100000):
        self.out_dir = out_dir
        ensure_dir_exists(self.out_dir)

        self.min_chunk_size = min_chunk_size
        self.cur_chunk_parts = []
        self.cur_chunk_size = 0
        self.cur_chunk_i = 0

    def append(self, part):
        self.cur_chunk_parts.append(part)
        self.cur_chunk_size += part.shape[0]

        if self.cur_chunk_size >= self.min_chunk_size:
            self._save_chunk()

    def _save_chunk(self):
        full_chunk = np.concatenate(self.cur_chunk_parts, axis=0).astype('float16')
        np.save(os.path.join(self.out_dir, 'chunk_{}.npy'.format(self.cur_chunk_i)), full_chunk)
        self.cur_chunk_parts = []
        self.cur_chunk_size = 0
        self.cur_chunk_i += 1

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self.cur_chunk_size > 0:
            self._save_chunk()
