#!/usr/bin/env python3.6

from youtube8m.utils.data_utils import shuffle_and_save_chunk

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
    parser.add_argument("output_path")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100000, help="The number of examples per one chunk")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_path, exist_ok=True)

    in_memory_fold_video_ids_chunks = [[] for _ in range(args.fold_count)]
    in_memory_fold_mean_rgb_chunks = [[] for _ in range(args.fold_count)]
    in_memory_fold_mean_audio_chunks = [[] for _ in range(args.fold_count)]
    in_memory_fold_labels_chunks = [[] for _ in range(args.fold_count)]

    fold_chunk_count = [0] * args.fold_count

    data_folders = os.listdir(args.input_path)
    random.shuffle(data_folders)

    next_fold_id = 0

    for data_folder_name in tqdm.tqdm(data_folders):
        video_ids_path = os.path.join(args.input_path, data_folder_name, "video_ids.npz")
        mean_rgb_path = os.path.join(args.input_path, data_folder_name, "mean_rgb.npz")
        mean_audio_path = os.path.join(args.input_path, data_folder_name, "mean_audio.npz")
        labels_path = os.path.join(args.input_path, data_folder_name, "labels.npz")

        video_ids_blob = np.load(video_ids_path)["arr_0"]
        mean_rgb_blob = np.load(mean_rgb_path)["arr_0"]
        mean_audio_blob = np.load(mean_audio_path)["arr_0"]
        labels_blob = np.load(labels_path)["arr_0"]

        for video_id, mean_rgb, mean_audio, label in zip(video_ids_blob, mean_rgb_blob, mean_audio_blob, labels_blob):
            in_memory_fold_video_ids_chunks[next_fold_id].append(video_id)
            in_memory_fold_mean_rgb_chunks[next_fold_id].append(mean_rgb)
            in_memory_fold_mean_audio_chunks[next_fold_id].append(mean_audio)
            in_memory_fold_labels_chunks[next_fold_id].append(label)

            if len(in_memory_fold_video_ids_chunks[next_fold_id]) == args.chunk_size:
                shuffle_and_save_chunk(
                    args.output_path,
                    next_fold_id,
                    fold_chunk_count[next_fold_id],
                    [in_memory_fold_video_ids_chunks[next_fold_id],
                     in_memory_fold_mean_rgb_chunks[next_fold_id],
                     in_memory_fold_mean_audio_chunks[next_fold_id],
                     in_memory_fold_labels_chunks[next_fold_id]],
                    DATA_FILE_NAMES)

                in_memory_fold_video_ids_chunks[next_fold_id].clear()
                in_memory_fold_mean_rgb_chunks[next_fold_id].clear()
                in_memory_fold_mean_audio_chunks[next_fold_id].clear()
                in_memory_fold_labels_chunks[next_fold_id].clear()

                fold_chunk_count[next_fold_id] += 1

            next_fold_id = (next_fold_id + 1) % args.fold_count

    for fold_id in range(args.fold_count):
        shuffle_and_save_chunk(
            args.output_path,
            fold_id,
            fold_chunk_count[fold_id],
            [in_memory_fold_video_ids_chunks[fold_id],
             in_memory_fold_mean_rgb_chunks[fold_id],
             in_memory_fold_mean_audio_chunks[fold_id],
             in_memory_fold_labels_chunks[fold_id]],
            DATA_FILE_NAMES)


if __name__ == "__main__":
    main()
