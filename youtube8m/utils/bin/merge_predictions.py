#!/usr/bin/env python3.6

from youtube8m.utils.io import BufferedPredictionsLoader
from youtube8m.utils.constants import FOLD_COUNT, VIDEO_LEVEL_FOLDS_PATH, VIDEO_LEVEL_TEST_PATH

import argparse
import numpy as np
import os
import tqdm


def merge_predictions(prediction_paths, output_path, id_chunks_path, is_training, chunk_size):
    os.makedirs(output_path, exist_ok=True)

    video_ids = []
    labels = []
    id_chunks_paths = os.listdir(id_chunks_path)

    for chunk_name_path in id_chunks_paths:
        video_ids_path = os.path.join(id_chunks_path, chunk_name_path, "video_ids")
        video_ids_chunk = np.load(video_ids_path)
        video_ids += video_ids_chunk.tolist()

        if is_training:
            labels_path = os.path.join(id_chunks_path, chunk_name_path, "labels")
            labels_chunk = np.load(labels_path)
            labels += labels_chunk.tolist()

    video_ids = np.array(video_ids)
    labels = np.array(labels)

    sort_order = np.argsort(video_ids)
    video_ids = video_ids[sort_order]

    if is_training:
        labels = labels[sort_order]

    prediction_buffers = []
    for path in prediction_paths:
        if not path:
            continue
        prediction_buffers.append(BufferedPredictionsLoader(path))

    in_memory_chunk_x = []
    in_memory_chunk_y = []
    in_memory_video_ids = []
    chunk_id = 0

    for i, video_id in enumerate(tqdm.tqdm(video_ids)):
        example = []
        for prediction_buffer in prediction_buffers:
            example.append(prediction_buffer.get_next())

        in_memory_chunk_x.append(np.array(example))
        in_memory_video_ids.append(video_id)

        if is_training:
            in_memory_chunk_y.append(labels[i])

        if len(in_memory_video_ids) == chunk_size:
            chunk_path = os.path.join(output_path, "chunk_{}".format(chunk_id))
            os.makedirs(chunk_path, exist_ok=True)

            with open(os.path.join(chunk_path, "video_ids"), "wb") as fout:
                np.save(fout, np.array(in_memory_video_ids))

            with open(os.path.join(chunk_path, "features"), "wb") as fout:
                np.save(fout, np.array(in_memory_chunk_x))

            if is_training:
                with open(os.path.join(chunk_path, "labels"), "wb") as fout:
                    np.save(fout, np.array(in_memory_chunk_y))

            chunk_id += 1
            in_memory_chunk_x = []
            in_memory_chunk_y = []
            in_memory_video_ids = []

    if in_memory_video_ids:
        chunk_path = os.path.join(output_path, "chunk_{}".format(chunk_id))
        os.makedirs(chunk_path, exist_ok=True)
        with open(os.path.join(chunk_path, "video_ids"), "wb") as fout:
            np.save(fout, np.array(in_memory_video_ids))

        with open(os.path.join(chunk_path, "features"), "wb") as fout:
            np.save(fout, np.array(in_memory_chunk_x))

        if is_training:
            with open(os.path.join(chunk_path, "labels"), "wb") as fout:
                np.save(fout, np.array(in_memory_chunk_y))


def main():
    parser = argparse.ArgumentParser(
        "This script merges predictions from different models and saves them in several chunks")
    parser.add_argument("model_paths_file")
    parser.add_argument("output_path")
    parser.add_argument("--chunk-size", type=int, default=10000)

    args = parser.parse_args()

    with open(args.model_paths_file) as fin:
        model_paths = fin.read().split("\n")

    for fold_id in range(FOLD_COUNT):
        prediction_paths = []
        for model_path in model_paths:
            prediction_path = os.path.join(model_path, "folds/fold_{}/predictions/sorted".format(fold_id))
            prediction_paths.append(prediction_path)

        merge_predictions(
            prediction_paths,
            os.path.join(args.output_path, "folds/fold_{}".format(fold_id)),
            "{}/fold_{}".format(VIDEO_LEVEL_FOLDS_PATH, fold_id),
            True,
            args.chunk_size)

    prediction_paths = []
    for model_path in model_paths:
        prediction_path = os.path.join(model_path, "test_averaged")
        prediction_paths.append(prediction_path)

    merge_predictions(
        prediction_paths,
        os.path.join(args.output_path, "test"),
        VIDEO_LEVEL_TEST_PATH,
        False,
        args.chunk_size)


if __name__ == "__main__":
    main()
