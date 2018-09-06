import os
import numpy as np
import pickle
import argparse
import glob
import scipy
from tqdm import tqdm
from youtube8m.video_level_nn_models.defaults import YOUTUBE8M_LABELS_N
from scipy.sparse import lil_matrix


def build_index(video_ids, features, scores):
    unique_ids = np.unique(video_ids)
    index = {unique_ids[i]: i for i in range(len(unique_ids))}
    sparse_matrix = lil_matrix((len(unique_ids), YOUTUBE8M_LABELS_N))
    for vid, lidx, score in tqdm(zip(video_ids, features, scores), total=len(video_ids)):
        sparse_matrix[index[vid], lidx] = score
    return index, sparse_matrix


def main(args):
    os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)
    folds_paths = sorted(glob.glob(os.path.join(args.path_with_folds, 'fold_*')))
    features_paths = sorted(glob.glob(os.path.join(args.path_with_features, 'folds', 'fold_*')))
    features, scores, video_ids = _load_train(folds_paths, features_paths)
    features = np.concatenate(features, axis=0)
    scores = np.concatenate(scores, axis=0)
    video_ids = np.concatenate(video_ids, axis=0)
    index, sparse_matrix = build_index(video_ids, features, scores)
    with open(args.output_filename, 'wb') as f:
        pickle.dump(index, f)
    scipy.sparse.save_npz(args.output_filename + '.matr', sparse_matrix.tocsr())


def _load_train(folds_paths, features_paths):
    features = []
    scores = []
    video_ids = []
    for fold_path, features_path in zip(folds_paths, features_paths):
        features.append(np.load(os.path.join(features_path, 'features'))[:, 0].astype(int))
        scores.append(np.load(os.path.join(fold_path, 'predictions.npy')))
        video_ids.append(np.load(os.path.join(fold_path, 'video_ids.npy')))
    return features, scores, video_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser("This script gathers all level 2 data to single video_id -> label score index")
    parser.add_argument("path_with_folds")
    parser.add_argument("path_with_features")
    parser.add_argument("output_filename")
    args = parser.parse_args()
    main(args)
