#!/usr/bin/env python3.6

from youtube8m.utils.tfrecord_utils import read_big_tfrecord

import argparse
import numpy as np
import os
import pickle
import tqdm

DATA_FILE_NAMES = [
    "video_ids",
    "rgb",
    "audio"
]


def save_chunk(output_path, chunk_id, arrays):
    numpy_arrays = [np.array(array) for array in arrays]

    output_path_chunk = os.path.join(output_path, "chunk_{}".format(chunk_id))
    os.makedirs(output_path_chunk, exist_ok=True)

    for array, file_name in zip(numpy_arrays, DATA_FILE_NAMES):
        file_path = os.path.join(output_path_chunk, file_name)
        with open(file_path, "wb") as fout:
            pickle.dump(array, fout)


def main():
    parser = argparse.ArgumentParser("This script converts test dataset into numpy and saves it in several chunks")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--chunk-size", type=int, default=1000, help="The number of examples per one chunk")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    in_memory_video_ids_chunks = []
    in_memory_rgb_chunks = []
    in_memory_audio_chunks = []

    chunk_count = 0

    files = os.listdir(args.input_path)
    files = list(filter(lambda name: name.endswith(".tfrecord"), files))

    for file_name in tqdm.tqdm(files):
        tfrecord_path = os.path.join(args.input_path, file_name)
        video_ids_blob, rgb_blob, audio_blob, _ = read_big_tfrecord(tfrecord_path, is_training=False)

        for video_id, rgb, audio in zip(video_ids_blob, rgb_blob, audio_blob):
            in_memory_video_ids_chunks.append(video_id)
            in_memory_rgb_chunks.append(rgb)
            in_memory_audio_chunks.append(audio)

            if len(in_memory_video_ids_chunks) == args.chunk_size:
                save_chunk(
                    args.output_path,
                    chunk_count,
                    [in_memory_video_ids_chunks,
                     in_memory_rgb_chunks,
                     in_memory_audio_chunks])

                in_memory_video_ids_chunks.clear()
                in_memory_rgb_chunks.clear()
                in_memory_audio_chunks.clear()

                chunk_count += 1

    if len(in_memory_video_ids_chunks) != 0:
        save_chunk(
            args.output_path,
            chunk_count,
            [in_memory_video_ids_chunks,
             in_memory_rgb_chunks,
             in_memory_audio_chunks])


if __name__ == "__main__":
    main()
