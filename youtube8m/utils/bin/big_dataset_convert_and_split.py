#!/usr/bin/env python3.6

from youtube8m.utils.data_utils import shuffle_and_save_chunk, get_video_id_to_fold_id_dict
from youtube8m.utils.tfrecord_utils import read_big_tfrecord

import argparse
import numpy as np
import os
import random
import tqdm


DATA_FILE_NAMES = [
    "video_ids",
    "rgb",
    "audio",
    "labels"
]


def main():
    parser = argparse.ArgumentParser("This script splits train dataset into folds and saves them in several chunks")
    parser.add_argument("input_path")
    parser.add_argument("fold_split_path")
    parser.add_argument("output_path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=1000, help="The number of examples per one chunk")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_path, exist_ok=True)

    video_id_to_fold_id = get_video_id_to_fold_id_dict(args.fold_split_path)
    fold_count = len(set(video_id_to_fold_id.values()))

    in_memory_fold_video_ids_chunks = [[] for _ in range(fold_count)]
    in_memory_fold_rgb_chunks = [[] for _ in range(fold_count)]
    in_memory_fold_audio_chunks = [[] for _ in range(fold_count)]
    in_memory_fold_labels_chunks = [[] for _ in range(fold_count)]

    fold_chunk_count = [0] * fold_count

    files = os.listdir(args.input_path)
    files = list(filter(lambda name: name.endswith(".tfrecord"), files))
    random.shuffle(files)

    for file_name in tqdm.tqdm(files):
        tfrecord_path = os.path.join(args.input_path, file_name)
        video_ids_blob, rgb_blob, audio_blob, labels_blob = read_big_tfrecord(tfrecord_path, is_training=True)

        for video_id, rgb, audio, label in zip(video_ids_blob, rgb_blob, audio_blob, labels_blob):
            fold_id = video_id_to_fold_id[video_id]
            in_memory_fold_video_ids_chunks[fold_id].append(video_id)
            in_memory_fold_rgb_chunks[fold_id].append(rgb)
            in_memory_fold_audio_chunks[fold_id].append(audio)
            in_memory_fold_labels_chunks[fold_id].append(label)

            if len(in_memory_fold_video_ids_chunks[fold_id]) == args.chunk_size:
                shuffle_and_save_chunk(
                    args.output_path,
                    fold_id,
                    fold_chunk_count[fold_id],
                    [in_memory_fold_video_ids_chunks[fold_id],
                     in_memory_fold_rgb_chunks[fold_id],
                     in_memory_fold_audio_chunks[fold_id],
                     in_memory_fold_labels_chunks[fold_id]],
                    DATA_FILE_NAMES,
                    use_pickle=True)

                in_memory_fold_video_ids_chunks[fold_id].clear()
                in_memory_fold_rgb_chunks[fold_id].clear()
                in_memory_fold_audio_chunks[fold_id].clear()
                in_memory_fold_labels_chunks[fold_id].clear()

                fold_chunk_count[fold_id] += 1

    for fold_id in range(fold_count):
        shuffle_and_save_chunk(
            args.output_path,
            fold_id,
            fold_chunk_count[fold_id],
            [in_memory_fold_video_ids_chunks[fold_id],
             in_memory_fold_rgb_chunks[fold_id],
             in_memory_fold_audio_chunks[fold_id],
             in_memory_fold_labels_chunks[fold_id]],
            DATA_FILE_NAMES,
            use_pickle=True)


if __name__ == "__main__":
    main()
