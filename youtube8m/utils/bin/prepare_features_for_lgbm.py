#!/usr/bin/env python3.6

from youtube8m.utils.io import sort_files_natural
from youtube8m.utils.constants import FOLD_COUNT

import argparse
import numpy as np
import os
import tqdm

TOP_LABELS_PER_MODEL = 20


def extract_features(prediction_path, output_path, is_training):
    os.makedirs(output_path, exist_ok=True)

    chunk_paths = sort_files_natural(os.listdir(prediction_path))

    X = []
    y = []
    candidates = []
    video_ids = []

    for chunk_id in tqdm.tqdm(chunk_paths):
        chunk_path = os.path.join(prediction_path, chunk_id)

        features = np.load(os.path.join(chunk_path, "features"))
        example_ids = np.load(os.path.join(chunk_path, "video_ids"))

        if is_training:
            labels = np.load(os.path.join(chunk_path, "labels"))

        for example_index in range(len(features)):
            labels_candidates = np.unique(
                np.argpartition(features[example_index], -TOP_LABELS_PER_MODEL)[:, -TOP_LABELS_PER_MODEL:].ravel())

            candidates_features = features[example_index][:, labels_candidates].transpose((1, 0)).astype("float32")

            for i in range(len(candidates_features)):
                lgbm_example = np.concatenate([[labels_candidates[i]], candidates_features[i]])
                X.append(lgbm_example)
                candidates.append(labels_candidates[i])
                video_ids.append(example_ids[example_index])

                if is_training:
                    label = int(labels_candidates[i] in labels[example_index])
                    y.append(label)

    with open(os.path.join(output_path, "features"), "wb") as fout:
        np.save(fout, np.array(X))

    with open(os.path.join(output_path, "candidates"), "wb") as fout:
        np.save(fout, np.array(candidates))

    with open(os.path.join(output_path, "video_ids"), "wb") as fout:
        np.save(fout, np.array(video_ids))

    if is_training:
        with open(os.path.join(output_path, "labels"), "wb") as fout:
            np.save(fout, np.array(y))

def main():
    parser = argparse.ArgumentParser("This script extracts features from predictions for training LGBM")
    parser.add_argument("merged_predictions_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    for fold_id in range(FOLD_COUNT):
        extract_features(
            os.path.join(args.merged_predictions_path, "folds/fold_{}".format(fold_id)),
            os.path.join(args.output_path, "folds/fold_{}".format(fold_id)),
            True)

    extract_features(
        os.path.join(args.merged_predictions_path, "test"),
        os.path.join(args.output_path, "test"),
        False)


if __name__ == "__main__":
    main()
