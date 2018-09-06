#!/usr/bin/env python3.6

from youtube8m.utils.io import BufferedPredictionsLoader
from youtube8m.utils.constants import FOLD_COUNT, VIDEO_LEVEL_TEST_PATH, FRAME_LEVEL_TEST_PATH

import argparse
import numpy as np
import os
import tqdm


def main():
    parser = argparse.ArgumentParser(
        "This script averages predictions from different models and saves them in several chunks")
    parser.add_argument("model_path")
    parser.add_argument("--frame-level", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=10000)

    args = parser.parse_args()

    output_path = os.path.join(args.model_path, "test_averaged")

    os.makedirs(output_path, exist_ok=True)

    video_ids = []

    id_chunks_path = VIDEO_LEVEL_TEST_PATH
    if args.frame_level:
        id_chunks_path = FRAME_LEVEL_TEST_PATH

    id_chunks_paths = os.listdir(id_chunks_path)

    for chunk_name_path in id_chunks_paths:
        video_ids_path = os.path.join(id_chunks_path, chunk_name_path, "video_ids")
        video_ids_chunk = np.load(video_ids_path)
        video_ids += video_ids_chunk.tolist()

    video_ids = np.array(video_ids)
    sort_order = np.argsort(video_ids)
    video_ids = video_ids[sort_order]

    prediction_buffers = []

    for fold_id in range(FOLD_COUNT):
        path = os.path.join(args.model_path, "folds", "fold_{}".format(fold_id), "test_predictions/sorted")
        prediction_buffers.append(BufferedPredictionsLoader(path))

    in_memory_chunk = []
    chunk_id = 0

    for _ in tqdm.trange(len(video_ids)):
        example = []
        for prediction_buffer in prediction_buffers:
            example.append(prediction_buffer.get_next())

        in_memory_chunk.append(np.array(example, dtype="float32").mean(axis=0))

        if len(in_memory_chunk) == args.chunk_size:
            chunk_path = os.path.join(output_path, "chunk_{}".format(chunk_id))

            with open(chunk_path, "wb") as fout:
                np.save(fout, np.array(in_memory_chunk))

            chunk_id += 1
            in_memory_chunk = []

    if in_memory_chunk:
        chunk_path = os.path.join(output_path, "chunk_{}".format(chunk_id))

        with open(chunk_path, "wb") as fout:
            np.save(fout, np.array(in_memory_chunk))


if __name__ == "__main__":
    main()
