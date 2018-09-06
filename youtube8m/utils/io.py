import glob
import importlib
import os
import pickle
import re
import shutil

import torch
import yaml

_METADATA_SUFFIX = '.meta.json'
_NUMBER_RE = re.compile(r'\d+')


def load_yaml(path):
    with open(path, 'r') as fin:
        return yaml.load(fin)


def save_yaml(data, path):
    with open(path, 'w') as fout:
        yaml.dump(data, fout)


def save_pickle(data, fname):
    with open(fname, 'wb') as fout:
        pickle.dump(data, fout)


def load_pickle(fname):
    with open(fname, 'rb') as fin:
        return pickle.load(fin)


def save_meta(meta_dict, out_file):
    with open(out_file + _METADATA_SUFFIX, 'w') as fout:
        yaml.dump(meta_dict, fout, indent=4)


def load_meta(base_fname):
    with open(base_fname + _METADATA_SUFFIX, 'r') as fin:
        return yaml.load(fin)


def get_full_class_name(obj):
    return obj.__module__ + "." + obj.__class__.__name__


def save_model(model, out_file, **extra_meta):
    meta = dict(extra_meta)
    meta['cls'] = get_full_class_name(model)
    save_meta(meta, out_file)
    torch.save(model.state_dict(), out_file)


def load_model(base_fname, model_kwargs):
    meta = load_meta(base_fname)
    model = load_class(meta['cls'])(**model_kwargs)
    model.load_state_dict(torch.load(base_fname))
    return model


def copy_model(src_base_fname, to_base_fname):
    for fname in glob.glob(src_base_fname + '*'):
        suffix = fname[len(src_base_fname):]
        shutil.copy2(fname, to_base_fname + suffix)


def load_class(full_name):
    module_name, class_name = full_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_obj_with_kwargs(config):
    return load_class(config['full_name'])(**config.get('kwargs', {}))


def ensure_dir_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def move_data_to_device(x, cuda, allow_using_any_cuda_device=False):
    if isinstance(x, torch.Tensor):
        if cuda and (x.device == torch.device("cpu") or not allow_using_any_cuda_device):
            x = x.cuda()
        if not cuda:
            x = x.cpu()
        return x
    elif isinstance(x, (list, tuple)):
        return [move_data_to_device(elem, cuda, allow_using_any_cuda_device) for elem in x]
    return x


def sort_files_natural(files):
    fnames_with_numbers = [(tuple(map(int, _NUMBER_RE.findall(fname))), fname) for fname in files]
    fnames_with_numbers.sort(key=lambda t: t[0])
    return [fname for _, fname in fnames_with_numbers]


def create_dirs(path):
    dirname = "/"
    splitted = path.strip(os.sep).split(os.sep)
    for subdir in splitted:
        dirname = os.path.join(dirname, subdir)
        try:
            os.mkdir(dirname)
        except FileExistsError:
            pass


class BufferedPredictionsLoader:
    def __init__(self, path):
        self._path = path
        self._chunks = sort_files_natural(os.listdir(path))
        self._next_chunk_id = 0
        self._buffer = None
        self._buffer_sample_id = None

        self._read_next_chunk()

    def _read_next_chunk(self):
        self._buffer = np.load(os.path.join(self._path, self._chunks[self._next_chunk_id]))
        self._buffer_sample_id = 0
        self._next_chunk_id += 1

    def get_next(self):
        if self._buffer_sample_id >= len(self._buffer):
            self._read_next_chunk()

        result = self._buffer[self._buffer_sample_id]
        self._buffer_sample_id += 1
        return result
